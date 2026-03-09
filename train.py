import datasets
import hydra
import transformers
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from src.models.utils import (
    get_model,
    get_condition_encoder,
    get_image_condition_encoder,
)
from src.data.collator import get_mesh_data_collator
from src.data.mesh import get_mesh_dataset, MeshProcessor
from src.utils.logging import JsonlLoggerCallback
from src.utils.trainer import CustomSFTTrainer, CustomSFTConfig
from src.utils.config import DataConfig, ModelConfig
from src.utils.ckpt import get_last_checkpoint
from src.utils.sig import SaveAndStopOnSignalCallback, install_sigusr1_handler

logger = get_logger(__name__)


OmegaConf.register_new_resolver("sub", lambda x, y: x - y)


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    install_sigusr1_handler()

    OmegaConf.resolve(cfg)

    accelerator = Accelerator()
    accelerator.print(OmegaConf.to_yaml(cfg))
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    local_model_path = model_cfg.local_path
    local_cond_model_path = model_cfg.local_cond_path

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_args = CustomSFTConfig(
        **OmegaConf.to_container(cfg.train.train_args, resolve=True),
        max_length=model_cfg.max_seq_length,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
        remove_unused_columns=False,
    )

    with accelerator.local_main_process_first():
        if model_cfg.img_cond:
            cond_encoder_img = get_image_condition_encoder(model_cfg)
        else:
            cond_encoder_img = None
        if model_cfg.cond:
            cond_encoder = get_condition_encoder(
                local_cond_model_path, model_cfg, cond_encoder_img=cond_encoder_img
            )
        else:
            cond_encoder = None
        model = get_model(
            local_model_path,
            model_cfg,
            cond_encoder=cond_encoder,
            cond_encoder_img=cond_encoder_img,
        )
        train_set, val_set, _ = get_mesh_dataset(data_cfg)

    sig_cb = SaveAndStopOnSignalCallback()
    trainer = CustomSFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=get_mesh_data_collator(data_cfg, model_cfg),
        processing_class=MeshProcessor(model_cfg),
        callbacks=[
            JsonlLoggerCallback(log_file_path=cfg.train.train_args.logging_dir),
            sig_cb,
        ],
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(train_args.output_dir))
    final_output_dir = Path(train_args.output_dir) / "final"
    if not sig_cb.signal_received:
        trainer.save_model(final_output_dir.as_posix())
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
