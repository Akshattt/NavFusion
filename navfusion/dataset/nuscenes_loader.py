from pathlib import Path
from nuscenes.nuscenes import NuScenes

# nuScenes dataset loader
class NuScenesLoader:
    def __init__(
            self,
            dataroot,
            version="v1.0-mini",
            verbose=True
    ):
        self.dataroot = Path(dataroot)

        if not self.dataroot.exists():
            raise ValueError(
                f"NuScenes dataset root path {self.dataroot} does not exist."
            )

        self.nusc = NuScenes(
            version=version,
            dataroot=str(self.dataroot),
            verbose=verbose
        )

    def get_scene(self, scene_index=0):
        return self.nusc.scene[scene_index]

    def get_sample_from_first_scene(self, scene_index=0):
        scene = self.get_scene(scene_index)

        sample = self.nusc.get(
            "sample",
            scene["first_sample_token"]
        )

        return sample

    # Sensor record access
    def get_sensor_record(self, sample, sensor_channel):
        if sensor_channel not in sample["data"]:
            raise KeyError(
                f"Sensor channel {sensor_channel} not found in sample data."
            )

        sample_data_token = sample["data"][sensor_channel]

        return self.nusc.get(
            "sample_data",
            sample_data_token
        )

    def get_sensor_file_path(self, sample_data_record):
        relative_filename = sample_data_record["filename"]

        full_path = self.dataroot / relative_filename

        return full_path

    def verify_sensor_file(self, sample_data_record):
        file_path = self.get_sensor_file_path(sample_data_record)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Sensor file {file_path} does not exist."
            )

        return file_path