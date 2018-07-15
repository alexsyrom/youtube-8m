import tensorflow as tf
import glob
import os

prefix = "/home/alexey/data/yt8m/video/train"
suffix = ".tfrecord"
files_count = 3844

files =  glob.glob("{}*{}".format(prefix, suffix)) 

filesSize = len(files)
report = open("report.txt", 'w')
read_indices = set()

for index, filename in enumerate(files):
    print('checking {}/{} {}'.format(index, filesSize, filename))
    file_index = int(filename[len(prefix):-len(suffix)])
    read_indices.add(file_index)
    try:
        for example in tf.python_io.tf_record_iterator(filename): 
            tf_example = tf.train.Example.FromString(example) 
    except Exception as ex:
        print("bad file {}".format(filename))
        print("bad file {} {}".format(filename, ex), file=report)

not_read_indices = set(range(files_count)) - read_indices
print("not_read_indices {}".format(not_read_indices), file=report)
