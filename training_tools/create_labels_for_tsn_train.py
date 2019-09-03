# coding: utf8
import argparse
import os
from pathlib import Path
import pandas as pd
from multiprocessing import Pool


VATIC_COLUMNS = (
    'local_id',
    'xmin',
    'ymin',
    'xmax',
    'ymax',
    'frame',
    'lost',
    'occuluded',
    'generated',
    'name',
    'action_1',
    'action_2',
    'action_3',
    'action_4'
)

ACTION_LIST = [
    'Calling',
    'Carrying',
    'Drinking',
    'Hand Shaking',
    'Hugging',
    'Lying',
    'Pushing/Pulling',
    'Reading',
    'Running',
    'Sitting',
    'Standing',
    'Walking',
    'NoAction'
]


def separate_skipped_frames(skipped_df, group_action_df):
    """
    example)
    skipped_df:
                local_id  xmin  ymin  xmax  ymax  frame  lost  occuluded  generated    name   action_1  action_2  action_3  action_4
        16374        90   539  1638   654  1852   2064     0          0          1  Person  Walking
        16378        90   548  1633   663  1846   2068     0          0          1  Person  Walking
    group_action_df:
            local_id  xmin  ymin  xmax  ymax  frame  lost  occuluded  generated    name   action
        16370        90   534  1644   649  1858   2060     0          0          0  Person  Walking
        16371        90   534  1641   649  1855   2061     0          0          1  Person  Walking
        16372        90   536  1641   652  1855   2062     0          0          1  Person  Walking
        16374        90   539  1638   654  1852   2064     0          0          1  Person  Walking
        16375        90   542  1636   657  1849   2065     0          0          1  Person  Walking
        16376        90   545  1636   660  1849   2066     0          0          1  Person  Walking
        16378        90   548  1633   663  1846   2068     0          0          1  Person  Walking
        16379        90   550  1630   666  1844   2069     0          0          1  Person  Walking

        ↓
            local_id  xmin  ymin  xmax  ymax  frame  lost  occuluded  generated    name   action
        16370        90   534  1644   649  1858   2060     0          0          0  Person  Walking
        16371        90   534  1641   649  1855   2061     0          0          1  Person  Walking
        16372        90   536  1641   652  1855   2062     0          0          1  Person  Walking

                local_id  xmin  ymin  xmax  ymax  frame  lost  occuluded  generated    name   action
        16374        90   539  1638   654  1852   2064     0          0          1  Person  Walking
        16375        90   542  1636   657  1849   2065     0          0          1  Person  Walking
        16376        90   545  1636   660  1849   2066     0          0          1  Person  Walking

                local_id  xmin  ymin  xmax  ymax  frame  lost  occuluded  generated    name   action
        16378        90   548  1633   663  1846   2068     0          0          1  Person  Walking
        16379        90   550  1630   666  1844   2069     0          0          1  Person  Walking
    """
    frame_list = skipped_df.frame
    prev_frame = None
    for frame in frame_list:
        if prev_frame is None:
            out_df = group_action_df[group_action_df.frame < frame]
        else:
            out_df = group_action_df[(prev_frame <= group_action_df.frame) & (
                group_action_df.frame < frame)]
        prev_frame = frame
        yield out_df


def output(output_dir, file_num, out_df, datalist_p, min_range, action_class_1, action_class_2, action_class_3, action_class_4):
    if len(out_df) < min_range:
        print('skip output because frame num {} < min_range {}'.format(
            len(out_df), min_range))
        return file_num
    p = write_separated_labels(output_dir, file_num, out_df)
    append_datalist(datalist_p, p, len(out_df), action_class_1, action_class_2, action_class_3, action_class_4)
    file_num += 1
    return file_num


def write_separated_labels(output_dir, output_file_name, out_df):
    p = Path(output_dir).joinpath('{}.txt'.format(output_file_name))

    # remove if already exists
    if p.exists():
        p.unlink()

    out_df = out_df.loc[:, ['local_id', 'xmin',
                            'ymin', 'xmax', 'ymax', 'frame']]
    out_df.to_csv(p, index=False, header=False, sep=' ')
    return p

def append_datalist(datalist_p, label_filepath, frame_num, action_class_1, action_class_2, action_class_3, action_class_4):
    '''
    append folloing formated text to datalist.txt:
    <labelpath> <frame_num> <action_class_1> <action_class_2> <action_class_3> <action_class_4>
    '''
    with datalist_p.open('a') as f:
        f.write('{} {} {} {} {} {}\n'.format(label_filepath, frame_num, action_class_1, action_class_2, action_class_3, action_class_4))


def main(label_dirpath, output_base_dir, min_range, worker=4):
    p = Path(label_dirpath)
    datalist_p = Path(output_base_dir).joinpath('datalist.txt')
    if datalist_p.exists():
        datalist_p.unlink()  # remove if already exists

    args = []
    for label_filepath in p.glob('*.txt'):
        args.append((label_filepath, datalist_p, min_range, output_base_dir))

    with Pool(worker) as pool:
        pool.starmap(process, args)


def process(label_filepath, datalist_p, min_range, output_base_dir):
    print('processing {}'.format(label_filepath.name))

    # make directory
    movie_name = label_filepath.stem  # /path/to/label/1.1.1.txt -> 1.1.1
    output_dir = os.path.join(output_base_dir, movie_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(label_filepath, sep=' ', names=VATIC_COLUMNS)

    # exclude ocluded and lost
    df = df[(df.lost == 0) & (df.occuluded == 0)]

    # ラベルリストに含まれない行動ラベルがあった場合、その行を除外する
    # exclude empty action label and not in ACTION_LIST
    df = df[df.action_1.isin(ACTION_LIST)]

    # ローカル人物単位で処理開始
    file_num = 0

    # dfの行動ラベルがNaNの場合、NoActionを代わりに代入する
    df.loc[df['action_2'].isnull(), 'action_2'] = 'NoAction'
    df.loc[df['action_3'].isnull(), 'action_3'] = 'NoAction'
    df.loc[df['action_4'].isnull(), 'action_4'] = 'NoAction'

    for _, gb_local_id in df.groupby('local_id'):
        #print("gb_local_id:{}".format(gb_local_id))
        for action, gb_action in gb_local_id.groupby(['action_1', 'action_2', 'action_3', 'action_4']):
            gb_action = gb_action.sort_values(by='frame')
            action_class_1 = ACTION_LIST.index(action[0])
            action_class_2 = -1 if action[1] == 'NoAction' else ACTION_LIST.index(action[1])
            action_class_3 = -1 if action[2] == 'NoAction' else ACTION_LIST.index(action[2])
            action_class_4 = -1 if action[3] == 'NoAction' else ACTION_LIST.index(action[3])

            # フレームが抜けた後の最初のフレームを取得
            skipped = gb_action[gb_action.frame.diff() > 1]
            if len(skipped) > 0:
                for out_df in separate_skipped_frames(skipped, gb_action):
                    write_separated_labels
                    file_num = output(
                        output_dir, file_num, out_df,
                        datalist_p, min_range, action_class_1,
                        action_class_2, action_class_3, action_class_4
                    )
            else:
                file_num = output(
                    output_dir, file_num, gb_action,
                    datalist_p, min_range, action_class_1,
                    action_class_2, action_class_3, action_class_4
                )


if __name__ == '__main__':

    description = '''
Description:
    与えられたラベルファイル（VATIC形式）を動線単位で分割し、
    テキストファイルに出力する

    example)
    input:
        label_dir = /path/to/labels/train
        output_dir = /path/to/labels-tsn/train

        /path/to/labels/train
            ├── 1.1.1.txt
            ├── 1.1.2.txt
            └── 1.1.3.txt

    output:
        /path/to/labels-tsn/train
            ├── 1.1.1
            │     ├── 0.txt
            │     ├── 1.txt
            │     └── 2.txt
            ├── 1.1.2
            │     ├── 0.txt
            │     ├── 1.txt
            │     └── 2.txt
            └── 1.1.3
                ├── 0.txt
                ├── 1.txt
                └── 2.txt

'''

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label_dir',
                        required=True,
                        help='ラベルファイルディレクトリパス')
    parser.add_argument('--output_dir',
                        required=True,
                        help='出力先のディレクトリパス')
    parser.add_argument('--min_range', type=int,
                        default=20,
                        help = 'exclude if separeted frames less than this')
    parser.add_argument('--worker', type=int, default=8)
    args = parser.parse_args()
    main(args.label_dir, args.output_dir, args.min_range, args.worker)
