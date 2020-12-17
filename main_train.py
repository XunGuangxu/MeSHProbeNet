import logging
import itertools
import torch
import os
import shutil
import torchtext
import argparse
from torchtext.data.pipeline import Pipeline
from meshprobenet import MeSHProbeNet
from dataset import Vocabulary, NumWordField, NumMeshField, NumJrnlField

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        default='./toy_data/train.tsv',
                        type=str,
                        help='The training data file of tsv format.')
    parser.add_argument('--dev_path',
                        default='./toy_data/validation.tsv',
                        type=str,
                        help='The validation data file of tsv format.')
    parser.add_argument('--expt_path',
                        default='./toy_data/save',
                        type=str,
                        help='The save path.')
    parser.add_argument('--src_vocab_pt',
                        default='./toy_data/vocab_w.txt',
                        type=str,
                        help='The text vocabulary file.')
    parser.add_argument('--jrnl_vocab_pt',
                        default='./toy_data/vocab_j.txt',
                        type=str,
                        help='The journal vocabulary file.')
    parser.add_argument('--tgt_vocab_pt',
                        default='./toy_data/vocab_m.txt',
                        type=str,
                        help='The mesh vocabulary file.')
    parser.add_argument('--src_max_len',
                        default=512,
                        type=int,
                        help='The maximum document length.')
    parser.add_argument('--batch_size',
                        default=340,
                        type=int,
                        help='The batch size.')
    parser.add_argument('--embed_dim',
                        default=355,
                        type=int,
                        help='The embedding dimension.')
    parser.add_argument('--hidden_dim',
                        default=355,
                        type=int,
                        help='The rnn hidden size.')
    parser.add_argument('--jrnl_dim',
                        default=100,
                        type=int,
                        help='The journal embedding dimension.')
    parser.add_argument('--n_probes',
                        default=5,
                        type=int,
                        help='The number of probes.')
    parser.add_argument('--n_layers',
                        default=2,
                        type=int,
                        help='The number of rnn layers.')
    parser.add_argument('--warmup_epochs',
                        default=1,
                        type=int,
                        help='The number of warmup epochs.')
    parser.add_argument('--num_epochs',
                        default=5,
                        type=int,
                        help='The number of training epochs.')
    parser.add_argument('--pad_id',
                        default=0,
                        type=int,
                        help='The padding id.')
    parser.add_argument('--learning_rate',
                        default=0.0025,
                        type=float,
                        help='The learning rate for Adam.')
    parser.add_argument('--weight_decay',
                        default=5e-10,
                        type=float,
                        help='The weight decay.')
    parser.add_argument('--do_eval',
                        action='store_true',
                        help='Whether to do the validation.')
    parser.add_argument('--do_save',
                        action='store_true',
                        help='Whether to save the model.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

    logger.info('device: {} n_gpu: {}'.format(device, n_gpu))
    logger.info('train data: {}'.format(args.train_path))
    logger.info('batch_size: {} embed_dim: {} n_probes: {} lr: {} weight_decay: {}'.format(args.batch_size, args.embed_dim, args.n_probes, args.learning_rate, args.weight_decay))

    src_vocab  = Vocabulary(args.src_vocab_pt)
    jrnl_vocab = Vocabulary(args.jrnl_vocab_pt)
    tgt_vocab  = Vocabulary(args.tgt_vocab_pt)
    src  = NumWordField(preprocessing=Pipeline(int), include_lengths=True, pad_id=args.pad_id, pre_max_len=args.src_max_len)
    jrnl = NumJrnlField(preprocessing=Pipeline(int), pad_id=args.pad_id)
    tgt  = NumMeshField(preprocessing=Pipeline(int), pad_id=args.pad_id, vocab_size=len(tgt_vocab.itos))
    train = torchtext.data.TabularDataset(path=args.train_path, format='tsv', fields=[('src', src), ('jrnl', jrnl), ('tgt', tgt)])
    dev   = torchtext.data.TabularDataset(path=args.dev_path, format='tsv', fields=[('src', src), ('jrnl', jrnl), ('tgt', tgt)])
    
    model = MeSHProbeNet(len(src_vocab.itos), args.embed_dim, args.hidden_dim, args.n_layers, args.n_probes, len(jrnl_vocab.itos), args.jrnl_dim, len(tgt_vocab.itos), n_gpu)
    
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Prepare data
    train_batch_iterator = torchtext.data.BucketIterator(dataset=train, batch_size=args.batch_size,
        sort=False, sort_within_batch=True, sort_key=lambda x: len(x.src), device=None, repeat=False)
    dev_batch_iterator = torchtext.data.BucketIterator(dataset=dev, batch_size=args.batch_size,
        sort=True, sort_key=lambda x: len(x.src), device=None, train=False)
    

    steps_per_epoch = len(train_batch_iterator)
    num_train_optimization_steps = len(train_batch_iterator) * args.num_epochs
    print_every = len(train_batch_iterator) // 10 + 1
    print_loss_total = 0
    epoch_loss_total = 0

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'self_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train))
    logger.info('  Batch size = %d', args.batch_size)
    logger.info('  Num steps = %d', num_train_optimization_steps)
    
    # learning rate linear scheduling with warmup
    pyramid_peak_step = steps_per_epoch * args.warmup_epochs
    pyramid_zero_step = steps_per_epoch * args.num_epochs
    def get_lr_rate(epoch, step):
        cur_total_steps = epoch * steps_per_epoch + step
        if cur_total_steps < pyramid_peak_step:
            return (cur_total_steps + 1) / pyramid_peak_step
        else:
            return (pyramid_zero_step - cur_total_steps) / (pyramid_zero_step - pyramid_peak_step)

    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_batch_iterator):
            lr_rate = get_lr_rate(epoch, step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * lr_rate
                
            input_variables, input_lengths  = getattr(batch, 'src')
            jrnl_variables = getattr(batch, 'jrnl')
            target_variables = getattr(batch, 'tgt')
    
            loss = model(input_variables, input_lengths, jrnl_variables, target_variables)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
    
            loss.backward()
            params = itertools.chain.from_iterable([group['params'] for group in optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, 1)
            optimizer.step()
            optimizer.zero_grad()        
            
            print_loss_total += loss.item()
            epoch_loss_total += loss.item()
            if step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                log_msg = 'Epoch: %d, Iteration: %.2f%%, Loss: %.6f, lr rate: %.2f' % (epoch, step / len(train_batch_iterator), print_loss_avg, lr_rate)
                logger.info(log_msg)
        epoch_loss_avg = epoch_loss_total / len(train_batch_iterator)
        epoch_loss_total = 0
        log_msg = 'Finished epoch %d: Train loss: %.6f' % (epoch, epoch_loss_avg)
        logger.info(log_msg)
        
        # validation
        if args.do_eval:
            model.eval()
            eval_loss_total = 0
            with torch.no_grad():
                for batch in dev_batch_iterator:
                    input_variables, input_lengths  = getattr(batch, 'src')
                    jrnl_variables = getattr(batch, 'jrnl')
                    target_variables = getattr(batch, 'tgt')
                    loss = model(input_variables, input_lengths, jrnl_variables, target_variables)
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    eval_loss_total += loss.item()
            eval_loss_avg = eval_loss_total / max(1, len(dev_batch_iterator))
            log_msg += ', Dev loss: %.6f' % (eval_loss_avg)
            logger.info(log_msg)
            model.train()
        
        # save the model
        if args.do_save:        
            path = os.path.join(args.expt_path, str(epoch))
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            torch.save(model, os.path.join(path, 'model.pt'))
        
        
if __name__ == '__main__':
    main()
