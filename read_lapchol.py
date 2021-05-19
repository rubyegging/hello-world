#%%
import os
import copy
import pandas as pd
from torch.utils.data import Dataset


def readLapcholAsDict(data_dir, anns_dir, incl_videos, sample_rate):
    """
    Reads Lapchol dataset as a dictionary.

    :param data_dir: relative path to directory with video folders ('video88', 'video95', ...)
    :type data_dir: str
    :param anns_dir: relative path to directory with phase annotation files ('video01-phase.txt, 'video02-phase.txt')
    :type anns_dir; str
    :param incl_videos: numbers of videos to be included in the dataset
    :type incl_videos: list of integers, for example [1, 2, 3] will include all frames from videos 1-3
    :param sample_rate: select one in {sample_rate} frames for dataset
    :type sample_rate: int
    """
    samples = {}
    idx = 0

    for video_nr in incl_videos:
        # Read label file of video and store in Dataframe
        anns_file = 'video' + f"{video_nr:02d}" + '.txt'
        anns_path = os.path.join(anns_dir, anns_file)

        if os.path.isfile(anns_path):
            anns_df = pd.read_csv(anns_path, delimiter='\t')
        else:
            print('Cannot find annotations for video ' + str(video_nr))
            continue

        # Read frame paths and store in dictionary, together with labels
        dir_name = 'video' + f"{video_nr:02d}"
        dir_path = os.path.join(data_dir, dir_name)

        if os.path.isdir(dir_path):
            for i, row in anns_df.iterrows():
                try:
                    frame_nr = int(row['frame'])
                # print(i)
                # Select only one frame in {sample_rate} number of frames
                    if frame_nr % sample_rate == 0:
                        G_A_Grade = row['G_A_Grade']    # change to Adhesions_label = row['A'] or G_A_Grade 
                    #frame_path = dir_path + '/video' + f"{video_nr:02d}" + '_' + str(frame_nr) + '.jpg'
                        frame_path = dir_path + '/frame-' + f"{frame_nr:06}" + '-000' + '.png'
                    # print('IDX', idx)
                    # print('Frame nr', frame_nr)

                        if os.path.isfile(frame_path):
                            samples[idx] = {
                            'frame_path': frame_path,
                            'frame_nr': frame_nr,
                            'video_nr': video_nr,
                            'G_A_Grade': G_A_Grade # change to 'Adhesions_label': Adhesions_label
                        }
                            idx += 1
                        else:
                            print('Cannot find frame', frame_nr, 'in video', video_nr)
                except:
                    print('Cannot convert float NaN to integer')


        else:
            print('Cannot find video frames for video ' + str(video_nr))

    return samples

# remove all samples with label: excl from lapchol_train
def filter_excl(samples):
    remove_samples_excl = []
    for sample in samples:
        if samples[sample]['G_A_Grade'] == 'excl':  # remove all samples with label: 0
            remove_samples_excl.append(sample)

    for sample in remove_samples_excl:
        del samples[sample]

    print(f'! Removed {len(remove_samples_excl)} samples with label excl')

    new_samples = {}
    new_idx = 0
    for sample in samples:
        new_samples[new_idx] = samples[sample]
        new_idx += 1
    samples = new_samples

    return samples

# remove all samples with label: 0 from lapchol_train
def filter_zeros(samples):
    remove_samples = []
    for sample in samples:
        if samples[sample]['G_A_Grade'] == 0: # remove all samples with label: 0
            remove_samples.append(sample)

    for sample in remove_samples:
        del samples[sample]
    print(f'! Removed {len(remove_samples)} samples with label 0')

    new_samples = {}
    new_idx = 0
    for sample in samples:
        new_samples[new_idx] = samples[sample]
        new_idx += 1
    samples = new_samples

    return samples


# change all 3's into 2's
def change_values(samples):
    samples_to_change = []
    for sample in samples:
        if samples[sample]['G_A_Grade'] == 3:
                samples_to_change.append(sample)

    for sample in samples_to_change:
        samples[sample]['G_A_Grade'] = 2
    print(f'! Changed {len(samples_to_change)} samples from 3 to 2')

    return samples

def all_ints(samples):
    float_to_int = []
    for sample in samples:
        float_to_int.append(sample)

    for sample in float_to_int:
        try:
            samples[sample]['G_A_Grade'] = int(samples[sample]['G_A_Grade'])
        except:
            print('This does not work for label excl')

    return samples


class LapcholDataset(Dataset):
    """
    Import Lapchol_dataset as a PyTorch Dataset.

    """

    def __init__(self, root_dir, incl_videos=[1], sample_rate=1, transform=None):
        """
        :param root_dir: relative directory of the Cholec80 dataset
        :type root_dir: str
        :param incl_videos: numbers of videos to be included in the dataset
        :type incl_videos: list of integers, for example [1, 2, 3] will include all frames from videos 1-3
        :param sample_rate: select one in {sample_rate} frames for dataset
        :type sample_rate: int
        """
        self.root_dir = root_dir
        self.incl_videos = incl_videos
        self.sample_rate = sample_rate
        self.transform = transform

        # Define subdirectories for video frames and phase annotations
        self.data_dir = os.path.join(self.root_dir, 'frames')
        self.anns_dir = os.path.join(self.root_dir, 'labels')

        # Load the Cholec80 as a dictionary of samples, incl. paths to frames and annotations
        self.samples = readLapcholAsDict(self.data_dir, self.anns_dir, self.incl_videos, self.sample_rate)
        self.samples = all_ints(self.samples) # tested
        self.samples = filter_excl(self.samples) # tested
        self.samples = filter_zeros(self.samples) # tested --> optional
        self.samples = change_values(self.samples) # tested --> optional

    def __len__(self):
        # Return length of samples dictionary
        return len(self.samples)

    def __getitem__(self, idx):
        # Return indexed item of samples dictionary
        item = copy.deepcopy(self.samples[idx])
        if self.transform:
            item = self.transform(item)
        return item
    
    def getdict(self):
        return self.samples

