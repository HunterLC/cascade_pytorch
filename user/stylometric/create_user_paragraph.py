# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 16:49
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : create_user_paragraph.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import csv
import os
import ijson.backends.python as ijson
from six import iteritems
from tqdm import tqdm

COMMENTS_DATASET_FILE_PATH = '../../data/comments.json'

USER_COMMENTS_FILE_PATH = '/train_balanced_user.csv'

# 定义输出目录
directory = './output'
if not os.path.exists(directory):
    os.makedirs(directory)


def main():
    users_comments_dict = collections.defaultdict(list)

    with tqdm(desc="Grouping comments by user", total=12704751) as progress_bar:
        inside_comment = False
        comment_text = None
        comment_username = None

        with open(COMMENTS_DATASET_FILE_PATH, 'rb') as file_:
            # As the JSON file is large (2.5GB) and everything is in one line, is better to read it as a stream,
            # using a SAX-like approach.
            for prefix, type_, value in ijson.parse(file_):
                if inside_comment:
                    if prefix.endswith('.text'):
                        comment_text = value
                    elif prefix.endswith('.author'):
                        comment_username = value
                    elif type_ == 'end_map':  # This assumes there are no nested maps inside the comment maps.
                        if comment_text and comment_username and comment_text != 'nan' \
                                and comment_username != '[deleted]':
                            users_comments_dict[comment_username].append(comment_text)

                        inside_comment = False
                        comment_text = None
                        comment_username = None

                        progress_bar.update()
                elif type_ == 'start_map' and prefix:
                    inside_comment = True

    with open(directory + USER_COMMENTS_FILE_PATH, 'w') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        writer.writerows((user, " <END> ".join(comments_texts))
                         for user, comments_texts in iteritems(users_comments_dict))


if __name__ == '__main__':
    main()
