from crop_pest_detection.data.datamodule import PestDataModule


def main():
    dm = PestDataModule(
        data_root="raw_data/agro_pest",
        num_classes=12,
        batch_size=2,
        num_workers=2,   # можно 0 для простоты
        pin_memory=False # на MPS смысла нет
    )
    dm.setup("fit")

    loader = dm.train_dataloader()
    images, targets = next(iter(loader))

    print("images:", len(images), images[0].shape, images[0].dtype, float(images[0].min()), float(images[0].max()))
    print("boxes:", targets[0]["boxes"].shape)
    print("labels:", targets[0]["labels"][:10])


if __name__ == "__main__":
    main()