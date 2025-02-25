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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        if hasattr(self.base_model, '_hook_before_iter'):
            self.base_model._hook_before_iter()
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