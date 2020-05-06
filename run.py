import argparse
from utils.utils import pp
import models
import os, sys, json
from oracle.oracle_gan.oracle_train import oracle_train
from real.real_gan.real_train import real_train
from real.real_gan.real_train_contin import real_train_traj
from utils.models.OracleLstm import OracleLstm
from oracle.oracle_gan.oracle_loader import OracleDataLoader
from real.real_gan.real_loader import RealDataLoader
from utils.text_process import text_precess
from subprocess import call

parser = argparse.ArgumentParser(description='Train and run a RmcGAN')
# Architecture
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator')
parser.add_argument('--g-architecture', default='rmc_att', type=str, help='Architecture for generator')
parser.add_argument('--d-architecture', default='rmc_att', type=str, help='Architecture for discriminator')
parser.add_argument('--gan-type', default='standard', type=str, help='Which type of GAN to use')
parser.add_argument('--hidden-dim', default=32, type=int, help='only used for OrcaleLstm and lstm_vanilla (generator)')
parser.add_argument('--sn', default=False, action='store_true', help='if using spectral norm')

# Training
parser.add_argument('--gsteps', default='1', type=int, help='How many training steps to use for generator')
parser.add_argument('--dsteps', default='5', type=int, help='How many training steps to use for discriminator')
parser.add_argument('--npre-epochs', default=150, type=int, help='Number of steps to run pre-training')
parser.add_argument('--nadv-steps', default=5000, type=int, help='Number of steps to run adversarial training')
parser.add_argument('--ntest', default=50, type=int, help='How often to run tests')
parser.add_argument('--d-lr', default=1e-4, type=float, help='Learning rate for the discriminator')
parser.add_argument('--gpre-lr', default=1e-2, type=float, help='Learning rate for the generator in pre-training')
parser.add_argument('--gadv-lr', default=1e-4, type=float, help='Learning rate for the generator in adv-training')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
parser.add_argument('--log-dir', default='./oracle/logs', type=str, help='Where to store log and checkpoint files')
parser.add_argument('--sample-dir', default='./oracle/samples', type=str, help='Where to put samples during training')
parser.add_argument('--optimizer', default='adam', type=str, help='training method')
parser.add_argument('--decay', default=False, action='store_true', help='if decaying learning rate')
parser.add_argument('--adapt', default='exp', type=str, help='temperature control policy: [no, lin, exp, log, sigmoid, quad, sqrt]')
parser.add_argument('--seed', default=123, type=int, help='for reproducing the results')
parser.add_argument('--temperature', default=1000, type=float, help='the largest temperature')
parser.add_argument('--datalimit', default=-1, type=int, help='data limit value')
parser.add_argument('--mc_samples', default=16, type=int, help='Number of MC samples for PG')
parser.add_argument('--pg_baseline_val', default=2.0, type=float, help='pg_baseline_val')
parser.add_argument('--no_excessive_debug', action='store_true', help='Do not save a lot of checkpoints')
parser.add_argument('--checkpt_every', default=500, type=int, help='Number of intermediate chkpts to save')

# evaluation
parser.add_argument('--nll-oracle', default=False, action='store_true', help='if using nll-oracle metric')
parser.add_argument('--nll-gen', default=False, action='store_true', help='if using nll-gen metric')
parser.add_argument('--bleu', default=False, action='store_true', help='if using bleu metric, [2,3,4,5]')
parser.add_argument('--selfbleu', default=False, action='store_true', help='if using selfbleu metric, [2,3,4,5]')
parser.add_argument('--doc-embsim', default=False, action='store_true', help='if using DocEmbSim metric')

# relational memory
parser.add_argument('--mem-slots', default=1, type=int, help="memory size")
parser.add_argument('--head-size', default=512, type=int, help="head size or memory size")
parser.add_argument('--num-heads', default=2, type=int, help="number of heads")

# Data
parser.add_argument('--dataset', default='oracle', type=str, help='[oracle, image_coco, emnlp_news, emnlp_news_small]')
parser.add_argument('--vocab-size', default=5000, type=int, help="vocabulary size")
parser.add_argument('--start-token', default=0, type=int, help="start token for a sentence")
parser.add_argument('--seq-len', default=20, type=int, help="sequence length: [20, 40]")
parser.add_argument('--num-sentences', default=10000, type=int, help="number of total sentences")
parser.add_argument('--gen-emb-dim', default=32, type=int, help="generator embedding dimension")
parser.add_argument('--dis-emb-dim', default=64, type=int, help="TOTAL discriminator embedding dimension")
parser.add_argument('--num-rep', default=64, type=int, help="number of discriminator embedded representations")
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored')
parser.add_argument('--load_saved_model', default='', type=str, help='Saved model location')
parser.add_argument('--saved_temperature', default=1.0, type=float, help='saved_temperature value')
parser.add_argument('--saved_global_step', default=0, type=int, help='saved_global_step value')
parser.add_argument('--saved_Wall_time', default=0.0, type=float, help='saved_Wall_time value')

#GAN-PG
parser.add_argument('--rl_only', default=False, action='store_true', help='')
parser.add_argument('--pg_baseline', default=False, action='store_true', help='')
parser.add_argument('--rl_use_multinomial', default=False, action='store_true', help='')
parser.add_argument('--rl_alpha', default=0.0, type=float, help="")  # 1, 0.0004
parser.add_argument('--rl_method', default=2, type=int, help="")  # 1, 2


def count_params(m_vars):
    total_parameters = 0
    # iterating over all variables
    for variable in m_vars:
        local_parameters = 1
        shape = variable.get_shape()  # getting shape of a variable
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
        total_parameters += local_parameters
    return total_parameters


def load_index_to_word_dict(itw_dict_path):
    index_to_word_dict = json.load(open(itw_dict_path))

    str_keys = [akey for akey in index_to_word_dict.keys()]
    index_to_word_dict = dict(zip(str_keys, index_to_word_dict.values()))
    return index_to_word_dict

def main():
    args = parser.parse_args()
    # pp.pprint(vars(args))
    config = vars(args)

    # train with different datasets
    if args.dataset == 'oracle':
        oracle_model = OracleLstm(num_vocabulary=args.vocab_size, batch_size=args.batch_size, emb_dim=args.gen_emb_dim,
                                  hidden_dim=args.hidden_dim, sequence_length=args.seq_len,
                                  start_token=args.start_token)
        oracle_loader = OracleDataLoader(args.batch_size, args.seq_len)
        gen_loader = OracleDataLoader(args.batch_size, args.seq_len)

        generator = models.get_generator(args.g_architecture, vocab_size=args.vocab_size, batch_size=args.batch_size,
                                         seq_len=args.seq_len, gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                         start_token=args.start_token)
        discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=args.seq_len,
                                                 vocab_size=args.vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)
        oracle_train(generator, discriminator, oracle_model, oracle_loader, gen_loader, config)

    elif args.dataset in ['image_coco', 'emnlp_news', 'emnlp_news_small']:
        data_file = os.path.join(args.data_dir, '{}.txt'.format(args.dataset))
        seq_len, vocab_size, word_index_dict, index_word_dict = text_precess(data_file)
        config['seq_len'] = seq_len
        config['vocab_size'] = vocab_size
        # print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

        oracle_loader = RealDataLoader(args.batch_size, args.seq_len)

        generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                         seq_len=seq_len, gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                         start_token=args.start_token)
        discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=seq_len,
                                                 vocab_size=vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)

        # print("gen params = ", count_params(generator.trainable_variables))
        # print("disc params = ", count_params(discriminator.trainable_variables))
        # sys.stdout.flush()

        load_model = False
        if config['load_saved_model'] != "":
            log_dir_path = os.path.dirname(config['load_saved_model'])
            config['log_dir'] = log_dir_path
            config['sample_dir'] = os.path.join(os.path.split(log_dir_path)[0], 'samples')
            index_word_dict = load_index_to_word_dict(os.path.join(config['log_dir'], "index_to_word_dict.json"))
            word_index_dict = {v: k for k, v in index_word_dict.items()}
            load_model=True
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'])
            json.dump(index_word_dict, open(os.path.join(config['log_dir'], "index_to_word_dict.json"), 'w'))
            json.dump(word_index_dict, open(os.path.join(config['log_dir'], "word_to_index_dict.json"), 'w'))

        pp.pprint(config)
        print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))
        sys.stdout.flush()
        real_train(generator, discriminator, oracle_loader, config, word_index_dict, index_word_dict, load_model=load_model)

        if args.dataset == "emnlp_news" or args.dataset == "emnlp_news_small":
            call(["python", 'bleu_post_training_emnlp.py', os.path.join(os.path.split(config['log_dir'])[0], 'samples'), 'na'], cwd=".")
        elif args.dataset == "image_coco":
            call(["python", 'bleu_post_training.py', os.path.join(os.path.split(config['log_dir'])[0], 'samples'), 'na'], cwd=".")

    elif args.dataset in ['ace0_small']:
        # data_file = os.path.join(args.data_dir, '{}.txt'.format(args.dataset))
        # seq_len, vocab_size, word_index_dict, index_word_dict = text_precess(data_file)
        seq_len = config['seq_len']
        vocab_size = config['vocab_size']
        # # print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

        # oracle_loader = RealDataLoader(args.batch_size, args.seq_len)

        generator = models.get_generator(args.g_architecture, vocab_size=config['vocab_size'], batch_size=args.batch_size,
                                         seq_len=config['seq_len'], gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                         start_token=args.start_token)
        discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=config['seq_len'],
                                                 vocab_size=config['vocab_size'], dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)

        # print("gen params = ", count_params(generator.trainable_variables))
        # print("disc params = ", count_params(discriminator.trainable_variables))
        # sys.stdout.flush()

        load_model = False
        if config['load_saved_model'] != "":
            log_dir_path = os.path.dirname(config['load_saved_model'])
            config['log_dir'] = log_dir_path
            config['sample_dir'] = os.path.join(os.path.split(log_dir_path)[0], 'samples')
            index_word_dict = load_index_to_word_dict(os.path.join(config['log_dir'], "index_to_word_dict.json"))
            word_index_dict = {v: k for k, v in index_word_dict.items()}
            load_model=True
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'])
            # json.dump(index_word_dict, open(os.path.join(config['log_dir'], "index_to_word_dict.json"), 'w'))
            # json.dump(word_index_dict, open(os.path.join(config['log_dir'], "word_to_index_dict.json"), 'w'))

        pp.pprint(config)
        print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))
        sys.stdout.flush()
        real_train_traj(generator, discriminator, None, config, None, None, load_model=load_model)

        # if args.dataset == "emnlp_news" or args.dataset == "emnlp_news_small":
        #     call(["python", 'bleu_post_training_emnlp.py', os.path.join(os.path.split(config['log_dir'])[0], 'samples'), 'na'], cwd=".")
        # elif args.dataset == "image_coco":
        #     call(["python", 'bleu_post_training.py', os.path.join(os.path.split(config['log_dir'])[0], 'samples'), 'na'], cwd=".")
    else:
        raise NotImplementedError('{}: unknown dataset!'.format(args.dataset))


if __name__ == '__main__':
    main()
