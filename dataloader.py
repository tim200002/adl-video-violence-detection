#import faulthandler
import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor

from torchvision.datasets.vision import VisionDataset

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
#from torchvision.vision import VisionDataset

#import torchvision.folder as folder
#import torchvision.video_utils as video_utils
#faulthandler.enable()

class Hockey(VisionDataset):
    '''    
    `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
    dataset.

    HMDB51 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the HMDB51 Dataset.
        annotation_path (str): Path to the folder containing the split files.
        frames_per_clip (int): Number of frames in a clip.
        step_between_clips (int): Number of frames between each clip.
        fold (int, optional): Which fold to use. Should be between 1 and 3.
        train (bool, optional): If ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip

    '''
    #data_url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    #splits = {
    #    "url": "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
    #    "md5": "15e67781e70dcfbdce2d7dbb9b3344b5",
    #}
 
    TRAIN_TAG = 0
    VALID_TAG = 1
    TEST_TAG = 2

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        #fold: int = 1,
        train: int = 0,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC",
    ) -> None:
        super().__init__(root)
        #if fold not in (1, 2, 3):
        #    raise ValueError(f"fold should be between 1 and 3, got {fold}")

        extensions = ("avi")
        
        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(
            self.root,
            class_to_idx,
            extensions,
        )
        print("class",self.classes )
        print("class",class_to_idx )
        #print("self.samples",self.samples) 
        video_paths = [path for (path, _) in self.samples]
        #print(video_paths)
        #exit()

        video_clips = VideoClips(
            video_paths,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )
        
        # we bookkeep the full version of video clips because we want to be able
        # to return the metadata of full version rather than the subset version of
        # video clips
        self.full_video_clips = video_clips
        print("length",self.full_video_clips.num_clips())
        #self.fold = fold
        self.train = train
        #self.indices = self._select_fold(video_paths, annotation_path, fold, train)
        #self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    #def _select_fold(self, video_list: List[str], annotations_dir: str, fold: int, train: bool) -> List[int]:
        #target_tag = self.TRAIN_TAG if train else self.TEST_TAG
        #split_pattern_name = f"*test_split{fold}.txt"
        #split_pattern_path = os.path.join(annotations_dir, split_pattern_name)
        #annotation_paths = glob.glob(split_pattern_path)
        #selected_files = set()
        #for filepath in annotation_paths:
        #    with open(filepath) as fid:
        #        lines = fid.readlines()
        #    for line in lines:
        #        video_filename, tag_string = line.split()
        #        tag = int(tag_string)
        #        if tag == target_tag:
        #            selected_files.add(video_filename)

        #indices = []
        #for video_index, video_path in enumerate(video_list):
        #    if os.path.basename(video_path) in selected_files:
        #        indices.append(video_index)

        #return indices

    def __len__(self) -> int:
        return self.full_video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, _, video_idx = self.full_video_clips.get_clip(idx)
        #sample_index = self.indices[video_idx]
        _, class_index = self.samples[video_idx]

        if self.transform is not None:
            video = self.transform(video)
        #print("getitem",video.size())
        return video, class_index

if __name__ == "__main__":
    num_frames = 5
    clip_steps = 1
    Bs_Train =5
    Bs_Test =5
    train_path = './data/HockeyFights/train/'
    print(train_path)
    train_dataset = Hockey(root=train_path, annotation_path='test_train_splits/', frames_per_clip=num_frames,step_between_clips = clip_steps,  train=0,transform=None, num_workers=1)
    i, j, k = train_dataset.__getitem__(0)
    print("size", i.size(),j.size(),k)
    #idx = torch.tensor([0,1,2,3])

    #video, audio, label = train_dataset[idx]

    #print(video.shape)
    #print(label)