import click


def _available_dataset_names() -> list[str]:
    from whar_datasets import WHARDatasetID

    if hasattr(WHARDatasetID, "__members__"):
        return list(WHARDatasetID.__members__.keys())
    return [str(item) for item in WHARDatasetID]


def _parse_dataset_id(dataset_name: str):
    from whar_datasets import WHARDatasetID

    dataset_name_normalized = dataset_name.strip()
    if not dataset_name_normalized:
        raise click.BadParameter("Dataset name cannot be empty.")

    if hasattr(WHARDatasetID, "__members__"):
        # Accept exact member name, case-insensitive member name, and values (if enum values are strings).
        members = WHARDatasetID.__members__
        if dataset_name_normalized in members:
            return members[dataset_name_normalized]
        for key, member in members.items():
            if key.lower() == dataset_name_normalized.lower():
                return member
            value = getattr(member, "value", None)
            if isinstance(value, str) and value.lower() == dataset_name_normalized.lower():
                return member

    for item in WHARDatasetID:
        if str(item).lower() == dataset_name_normalized.lower():
            return item
        value = getattr(item, "value", None)
        if isinstance(value, str) and value.lower() == dataset_name_normalized.lower():
            return item

    available = ", ".join(_available_dataset_names())
    raise click.BadParameter(f"Unknown dataset '{dataset_name}'. Options: {available}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """TinyHARFoundation command line."""


@cli.command("create-imu-dataset")
@click.argument("dataset_name")
def create_imu_dataset(dataset_name: str) -> None:
    """Create/cache an IMU dataset by name."""
    dataset_id = _parse_dataset_id(dataset_name)
    click.echo(f"Creating IMU dataset for '{dataset_id}'...")
    from src.Data.IMULocationDataset import IMULocationDataset
    IMULocationDataset(dataset_id)    
    click.echo("Done.")


if __name__ == "__main__":
    cli()
