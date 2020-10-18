import torch
import pytorch_lightning as pl
import datasets
import transformers
from absl import app, flags, logging

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
FLAGS = flags.FLAGS


class IMDBSentimentClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        # Single entry in train_ds is {'label': int, 'text': str}
        tokenizer = transformers.BertTokenizerFast.from_pretrained(FLAGS.model)
        # import IPython ; IPython.embed() ; exit(1)

        # So basically for each entry we add 'input_ids' field with tokenized
        # text so BERT can eat it and be healthy
        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                x['text'],
                max_length = 32,
                #padding = 'longest'
                pad_to_max_length = True
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

    def forward(self, input_ids):
        # First we construct an attention mask which ignores padding
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()

        """
        So depending on what we return in these functions
        pytorch lightning automatically does some things, 
        for example, just because we return a dict with 
        a 'loss' entry here, it will automatically run
        backpropagation based on this loss.
        Similarly with 'acc' in validation step and so on.
        """
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    # def on_validation_epoch_end(self, outputs):
    #     # This is just for the logger pretty much
    #     loss = torch.cat([o['loss'] for o in outputs], 0).mean() # jesus
    #     acc = torch.cat([o['acc'] for o in outputs], 0).mean()
    #     out = {'loss': loss, 'val_acc':acc}
    #     return {**out, 'log': out}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size = FLAGS.batch_size,
            drop_last = True,
            shuffle = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size = FLAGS.batch_size,
            drop_last = False,
            shuffle = False
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
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=1, max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        #logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0)
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)
