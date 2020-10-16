import torch
import pytorch_lightning as pl
import datasets
import transformers
from absl import app, flags, logging

flags.DEFINE_boolean('debug', True, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
FLAGS = flags.FLAGS

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # Single entry in train_ds is {'label': int, 'text': str}
        tokenizer = transformers.BertTokenizerFast.from_pretrained(FLAGS.model)
        # import IPython ; IPython.embed() ; exit(1)
        # So basically for each entry we add 'input_ids' field with tokenized review
        # so BERT can eat it and be healthy
        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                x['text'],
                max_length = 32,
                padding = 'longest' # possible not-goodness?
            )
            return x

        def _prepare_ds(split):
            ds = datasets.load_dataset('imdb', split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else "5%"}]')

            # This map() is not the standard python map but some parallelized huggingface version I think
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds = _prepare_ds("train")
        self.test_ds = _prepare_ds("test")

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size = FLAGS.batch_size,
            drop_last = True,
            shuffle = True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr = FLAGS.lr,
            momentum=FLAGS.momentum,
        )
        return optimizer


def main(_):
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(default_root_dir='logs', gpus=1, max_epochs=FLAGS.epochs, fast_dev_run=FLAGS.debug)
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)