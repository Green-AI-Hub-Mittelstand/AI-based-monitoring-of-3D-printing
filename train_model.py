import torch
from lightning import Trainer
from Model import ErrorDetectionModel
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Train a new model (run "download_images.py" first)

    num_classes = 10
    json_path = "images/output.json"  # path to annotation file
    image_path = "images/Images/"  # path to images
    batch_size = 16  # Set lower in case of Out of Memory Error
    n_epochs = 50  # Number of training epochs

    model = ErrorDetectionModel(10, json_path, image_path, batch_size)

    chpt_path = 'checkpoints'  # The top two models are sved to this path
    checkpoint_callback = ModelCheckpoint(dirpath=chpt_path, save_top_k=2, monitor="fscore")
    trainer = Trainer(devices=1, accelerator="gpu", check_val_every_n_epoch=1, num_sanity_val_steps=0,
                      max_epochs=int(n_epochs), default_root_dir=chpt_path, callbacks=[checkpoint_callback],
                      precision="bf16-mixed", log_every_n_steps=3,
                      enable_progress_bar=True, logger=False, enable_checkpointing=False)

    trainer.fit(model)

    # Tests the model with the test set and saves it
    print('Test with Test set')
    trainer.test()
    print(trainer.logged_metrics)
    torch.save(model.model.state_dict(), 'model_' + str(trainer.logged_metrics['fscore']) + '.pth')
