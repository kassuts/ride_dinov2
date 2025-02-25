import torch
import types
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
        
        # Move the base model to the device first
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = self.base_model.to(device)

        # Setup LoRA configuration
        lora_config = config['lora']
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', [
                "qkv",
                "proj",
                "fc1",
                "fc2"
            ]),
            bias=lora_config.get('bias', 'none'),
        )

        # Apply LoRA to the model
        model = get_peft_model(model, peft_config)
        
        # Explicitly move all model parts to the device
        model = model.to(device)
        
        # Ensure patch_embed is explicitly moved to device
        if hasattr(model, 'patch_embed'):
            model.patch_embed = model.patch_embed.to(device)
            if hasattr(model.patch_embed, 'proj'):
                model.patch_embed.proj = model.patch_embed.proj.to(device)
                
        # If the model has a backbone with patch_embed
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'patch_embed'):
            model.backbone.patch_embed = model.backbone.patch_embed.to(device)
            if hasattr(model.backbone.patch_embed, 'proj'):
                model.backbone.patch_embed.proj = model.backbone.patch_embed.proj.to(device)
        
        # Define a custom forward method for the wrapped model
        original_forward = model.forward
        def new_forward(self, x):
            # Ensure input is on the right device
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
            
            if hasattr(self, 'base_model'):
                # Pass directly to backbone for vision models
                if hasattr(self.base_model, 'backbone'):
                    return self.base_model(x)
                else:
                    return self.base_model(x)
            else:
                # Fallback
                return original_forward(x)
        
        # Bind the new forward method to the model
        model.forward = types.MethodType(new_forward, model)
        
        # Print trainable parameters
        print("\nTrainable parameters:")
        model.print_trainable_parameters()
        
        # Create optimizer after LoRA adaptation
        optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, 
                        combiner, finetuning_combiner, valid_data_loader, val_criterion,
                        lr_scheduler, len_epoch, save_imgs)
        
        # Double-check device placement for all model components
        self._ensure_model_on_device()
    
    def _ensure_model_on_device(self):
        """Make sure all model components are on the same device"""
        if self.device.type != 'cpu':
            print(f"Moving all model components to {self.device}...")
            
            # First move the entire model to device
            self.model = self.model.to(self.device)
            
            # Then ensure the base model is also on the device
            if hasattr(self, 'base_model'):
                self.base_model = self.base_model.to(self.device)
                
                # Explicitly move backbone if it exists
                if hasattr(self.base_model, 'backbone'):
                    self.base_model.backbone = self.base_model.backbone.to(self.device)
                    
                    # Explicitly move patch embed
                    if hasattr(self.base_model.backbone, 'patch_embed'):
                        self.base_model.backbone.patch_embed = self.base_model.backbone.patch_embed.to(self.device)
                        
                        # Explicitly move the projection layer
                        if hasattr(self.base_model.backbone.patch_embed, 'proj'):
                            self.base_model.backbone.patch_embed.proj = self.base_model.backbone.patch_embed.proj.to(self.device)
            
            # For LoRA adapters, check each parameter and buffer explicitly
            for name, module in self.model.named_modules():
                module.to(self.device)
                
            print(f"All model components moved to {self.device}")

    def _train_epoch(self, epoch):
        """Training logic for an epoch"""
        self.model.train()
        if hasattr(self.base_model, '_hook_before_iter'):
            self.base_model._hook_before_iter()
            
        # Double-check device placement at the beginning of each epoch
        self._ensure_model_on_device()
            
        return super()._train_epoch(epoch)

    def _save_checkpoint(self, epoch, save_best=False):
        """Saving LoRA checkpoints"""
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