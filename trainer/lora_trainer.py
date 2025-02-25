import torch
from peft import get_peft_model, LoraConfig, TaskType
from .trainer import Trainer


class LoRATrainer(Trainer):
    """
    Trainer class that implements LoRA for efficient finetuning
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, combiner,
                 finetuning_combiner=None, valid_data_loader=None, val_criterion=None,
                 lr_scheduler=None, len_epoch=None, save_imgs=False):
        
        # Store original model
        self.base_model = model

        # Setup LoRA configuration
        lora_config = config['lora']
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', [
                "attn.qkv",
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2"
            ]),
            bias=lora_config.get('bias', 'none'),
        )

        # Apply LoRA to the model
        model = get_peft_model(model, peft_config)
        
        # Override PEFT's forward to handle vision inputs
        def new_forward(self, x):
            # Ensure input is on the same device as model
            device = next(self.parameters()).device
            if x.device != device:
                x = x.to(device)
            
            # Ensure we're accessing the correct model
            if hasattr(self.base_model, 'model'):
                return self.base_model.model(x)
            else:
                return self.base_model(x)
        
        model.forward = new_forward.__get__(model, type(model))
        
        # Print trainable parameters to verify LoRA application
        print("\nTrainable parameters:")
        model.print_trainable_parameters()

        # Create optimizer after LoRA adaptation
        optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, 
                        combiner, finetuning_combiner, valid_data_loader, val_criterion,
                        lr_scheduler, len_epoch, save_imgs)
        
        # Make sure all parts of the model are on the same device
        self._ensure_model_on_device()

    def _ensure_model_on_device(self):
        """Make sure all model components are on the same device"""
        if self.device.type != 'cpu':
            print(f"Moving model to {self.device}...")
            
            # First move the entire model to device
            self.model = self.model.to(self.device)
            
            # Then ensure the base model is also on the device
            if hasattr(self, 'base_model'):
                self.base_model = self.base_model.to(self.device)
            
            # For LoRA adapters, we need to check each parameter and buffer explicitly
            for name, param in self.model.named_parameters():
                if param.device.type != self.device.type:
                    print(f"Moving parameter {name} from {param.device} to {self.device}")
                    param.data = param.data.to(self.device)
            
            for name, buffer in self.model.named_buffers():
                if buffer.device.type != self.device.type:
                    print(f"Moving buffer {name} from {buffer.device} to {self.device}")
                    buffer.data = buffer.data.to(self.device)
            
            # Handle specific modules separately (like patch embedding)
            for module in self.model.modules():
                if hasattr(module, 'proj') and hasattr(module.proj, 'weight'):
                    if module.proj.weight.device.type != self.device.type:
                        print(f"Moving proj weights from {module.proj.weight.device} to {self.device}")
                        module.proj.weight.data = module.proj.weight.data.to(self.device)
                        if hasattr(module.proj, 'bias') and module.proj.bias is not None:
                            module.proj.bias.data = module.proj.bias.data.to(self.device)
            
            print(f"Verified all model components are on {self.device}")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        if hasattr(self.base_model, '_hook_before_iter'):
            self.base_model._hook_before_iter()
            
        # Double-check device placement at the beginning of each epoch
        self._ensure_model_on_device()
            
        return super()._train_epoch(epoch)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving LoRA checkpoints
        """
        arch = type(self.base_model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best_lora.pth')
            torch.save(state, best_path)
            self.logger.info(f'Saving current best LoRA: model_best_lora.pth ...')
        else:
            filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}_lora.pth')
            torch.save(state, filename)
            self.logger.info(f'Saving checkpoint: {filename} ...') 