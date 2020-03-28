change some code in master
# DSSM截取运行的前半部分
import torch
import matchzoo as mz
import DSSM_preprocess
add_in_test = DSSM_preprocess.DSSM_preprocess(1)
d = DSSM_preprocess.DSSM_preprocess(1)
t = d.preprocess(6)
print(t)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# define a task
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]

# prepare input data
train_pack = mz.datasets.wiki_qa.load_data('train', task=ranking_task)  # task属性去掉，matchzoo.data_pack.data_pack.DataPack，改参数
valid_pack = mz.datasets.wiki_qa.load_data('dev', task=ranking_task)
print(train_pack)
# preprocess the input data
preprocessor = mz.models.DSSM.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)  # matchzoo.data_pack.data_pack.DataPack

# generate pair-wise training data, point valid set
trainset = mz.dataloader.Dataset(  # matchzoo.dataloader.dataset.Dataset
    data_pack=train_processed,
    mode='pair',
    num_dup=1,  # duplication
    num_neg=4
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point'
)

