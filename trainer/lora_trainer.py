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
        print(f"Moving model to {device}...")
        
        # Explicitly move all model components to device BEFORE LoRA adaptation
        def ensure_device_recursive(module, device):
            """Recursively move all module parameters and buffers to device"""
            for param in module.parameters(recurse=False):
                if param.device != device:
                    print(f"Moving parameter of shape {param.shape} from {param.device} to {device}")
                    param.data = param.data.to(device)
            
            for buf in module.buffers(recurse=False):
                if buf.device != device:
                    print(f"Moving buffer of shape {buf.shape} from {buf.device} to {device}")
                    buf.data = buf.data.to(device)
            
            # Move module itself
            module.to(device)
            
            # Recursively handle children
            for name, child in module.named_children():
                ensure_device_recursive(child, device)
            
            return module
        
        # Move ALL components to device before LoRA adaptation
        self.base_model = ensure_device_recursive(self.base_model, device)
        
        # Verify all components are on the correct device
        on_device = True
        for name, param in self.base_model.named_parameters():
            if param.device != device:
                print(f"ERROR: Parameter {name} still on {param.device} instead of {device}")
                on_device = False
        
        if on_device:
            print(f"Verified all model components are on {device}")
        
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
        
        # Move LoRA model to device
        model = ensure_device_recursive(model, device)
        
        # Create a custom forward method to bypass PEFT's text-specific logic
        def custom_forward(self, x):
            # Ensure input tensor is on the correct device
            if x.device != device:
                x = x.to(device)
            
            # Call the base model's forward directly
            return self.base_model(x)
        
        # Bind the new forward method to the model
        model.forward = types.MethodType(custom_forward, model)
        
        # Print trainable parameters
        print("\nTrainable parameters:")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.shape}")
        
        # Create optimizer after LoRA adaptation
        optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
        
        # Initialize parent class
        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, 
                         combiner, finetuning_combiner, valid_data_loader, val_criterion,
                         lr_scheduler, len_epoch, save_imgs)
        
        # Final device check after parent initialization
        print(f"Moving model to {self.device}...")
        self.model = ensure_device_recursive(self.model, self.device)
        
        # Verify one last time
        on_device = True
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"ERROR: After initialization, parameter {name} still on {param.device}")
                on_device = False
        
        if on_device:
            print(f"Verified all model components are on {self.device}")
        
    def _train_epoch(self, epoch):
        """Training logic for an epoch"""
        # Double check device placement before training
        device = self.device
        
        def check_device_recursive(module, prefix=""):
            """Check if all parameters and buffers are on the expected device"""
            all_on_device = True
            
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                if param.device != device:
                    print(f"Moving {full_name} from {param.device} to {device}")
                    param.data = param.data.to(device)
                    all_on_device = False
            
            for name, buf in module.named_buffers(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                if buf.device != device:
                    print(f"Moving buffer {full_name} from {buf.device} to {device}")
                    buf.data = buf.data.to(device)
                    all_on_device = False
            
            # Recursively check children
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                child_result = check_device_recursive(child, child_prefix)
                all_on_device = all_on_device and child_result
            
            return all_on_device
        
        # Check and fix device placement for all modules
        all_on_device = check_device_recursive(self.model)
        if all_on_device:
            print(f"All model parameters confirmed on {device} before epoch {epoch}")
        
        self.model.train()
        if hasattr(self.base_model, '_hook_before_iter'):
            self.base_model._hook_before_iter()
        
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