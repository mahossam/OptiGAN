import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time, sys
from utils.metrics.Nll import Nll
from utils.metrics.DocEmbSim import DocEmbSim
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu
from utils.utils import *
from utils.ops import gradient_penalty
import json
import pandas as pd
from utils import utils_ace0
from real.real_gan.trajc_data_loader import Trajc_DataLoader
from datetime import datetime

EPS = 1e-10

def save_checkpoint(saver, sess, save_folder, iter_id):
    # timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
    model_file_name = save_folder  # + iter_id + ".ckpt"
    saver.save(sess, model_file_name, iter_id)

def load_checkpoint(self, sess, location):
    self.saver.restore(sess, location)

# A function to initiate the graph and train the networks
def real_train_traj(generator, discriminator, oracle_loader, config, word_index_dict, index_word_dict, load_model=False):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    data_dir = config['data_dir']
    dataset = config['dataset']
    log_dir = config['log_dir']
    sample_dir = config['sample_dir']
    npre_epochs = config['npre_epochs']
    nadv_steps = config['nadv_steps']
    # temper = config['temperature']
    # adapt = config['adapt']

    # filename
    # oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(dataset))
    # oracle_file_no_padding = os.path.join(sample_dir, 'oracle_no_padding_{}.txt'.format(dataset))
    # gen_file = os.path.join(sample_dir, 'generator.txt')
    # gen_text_file = os.path.join(sample_dir, 'generator_text.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan.csv')
    # data_file = os.path.join(data_dir, '{}.txt'.format(dataset))
    # if dataset == 'image_coco':
    #     test_file = os.path.join(data_dir, 'testdata/test_coco.txt')
    # elif dataset == 'emnlp_news' or dataset == 'emnlp_news_small':
    #     test_file = os.path.join(data_dir, 'testdata/test_emnlp.txt')
    # else:
    #     raise NotImplementedError('Unknown dataset!')

    loader = None

    loader = Trajc_DataLoader(batch_size, seq_len)
    train, test, data_size, train_size, test_size, input_size, _, _ = loader.create_batches(data_dir, dataset)
    assert input_size == config['gen_emb_dim']
    output_size = input_size  # The unconditional case
    vocab_size = output_size

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    x_real = tf.placeholder(tf.float32, [batch_size, seq_len, input_size], name="x_real")  # tokens of oracle sequences

    # temperature = tf.Variable(config['saved_temperature'], trainable=False, name='temperature')

    # x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    disc_x_real = tf.placeholder(tf.float32, [batch_size, seq_len, input_size], name="disc_x_real")
    # assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    if '_pg' in config['g_architecture'] and '_pg' in config['d_architecture']:
        x_fake_onehot_appr, x_fake, x_fake_gaussian, g_pretrain_loss, gen_o, given_num, r_x, r_gen_x, r_gen_x_sample = generator(x_real=x_real, temperature=-9999.9)
    else:
        x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o = generator(x_real=x_real, temperature=-9999.9)

    d_out_real = discriminator(x_onehot=disc_x_real)
    d_out_fake = discriminator(x_onehot=x_fake_onehot_appr)

    # GAN / Divergence type

    rl_summaries = [tf.summary.scalar("just_a_dummy", tf.constant(0, tf.int32, shape=()))]
    rewards, initial_samples_for_rewards = None, None
    if '_pg' in config['g_architecture'] and '_pg' in config['d_architecture']:
        rewards = tf.placeholder(tf.float32, shape=[batch_size, seq_len], name="rewards")
        initial_samples_for_rewards = tf.placeholder(tf.float32, shape=[batch_size, seq_len],
                                                     name="initial_samples_for_rewards")
        # x_fake_for_rewards = x_fake
        # x_fake_for_rewards = x_fake_gaussian
        rl_summaries = [tf.summary.histogram("RL/rewards", rewards)]

    log_pg, g_loss, d_loss, reinforce_loss = get_losses(d_out_real, d_out_fake, disc_x_real, x_fake_onehot_appr,
                                                    gen_o, discriminator, config, rewards, initial_samples_for_rewards)

    # Global step
    global_step = tf.Variable(config['saved_global_step'], trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_pretrain_op, g_train_op, d_train_op = get_train_ops(config, g_pretrain_loss, g_loss, d_loss,
                                                          log_pg, -9999.9, global_step)

    # Record wall clock time
    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(config['saved_Wall_time'], trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    # temp_var = tf.placeholder(tf.float32)
    # update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('loss/discriminator', d_loss),
        tf.summary.scalar('loss/g_loss', g_loss),
        # tf.summary.scalar('loss/log_pg', log_pg),
        tf.summary.scalar('loss/Wall_clock_time', Wall_clock_time),
        # tf.summary.scalar('loss/temperature', temperature),
        tf.summary.scalar('loss/reinforce_loss', reinforce_loss)
    ]
    loss_summary_op = tf.summary.merge(loss_summaries + rl_summaries)

    # Metric Summaries
    metrics_pl, metric_summary_op = get_metric_summary_op(config)

    # ------------- initial the graph --------------
    with init_sess() as sess:
        log = open(csv_file, 'a')  # log = open(csv_file, 'w')

        saver = tf.train.Saver(max_to_keep=2)
        if load_model:
            saver.restore(sess, config['load_saved_model'])
        saver_intermediate = tf.train.Saver(max_to_keep=(nadv_steps//500))
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)

        # generate oracle data and create batches
        # eof_code = len(index_word_dict)
        # get_oracle_file(data_file, oracle_file, seq_len, word_index_dict)  # index_word_dict = get_oracle_file(data_file, oracle_file, seq_len)
        # save_oracle_file_no_padding(data_file, oracle_file_no_padding, seq_len, word_index_dict)
        # oracle_loader.create_batches(oracle_file, oracle_file_no_padding, config['datalimit'])

        # metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, g_pretrain_loss, x_real, sess)  #TODO
        metrics = get_metrics(config, oracle_loader, None, None, g_pretrain_loss, x_real, sess)  #TODO

        # log_header_names = "epoch, " + ", ".join([f", {metric.get_name()}" for metric in metrics])  #TODO
        # log.write(log_header_names)
        # log.write('\n')
        if load_model == False:
            print('Start pre-training...')
            for epoch in range(npre_epochs):
                # pre-training
                g_pretrain_loss_np = pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, oracle_loader)

                # Test
                ntest_pre = 10
                if np.mod(epoch, ntest_pre) == 0:
                    # generate fake data and create batches
                    # gen_save_file = os.path.join(sample_dir, 'pre_samples_{:05d}.txt'.format(epoch))
                    # generate_samples(sess, x_fake, batch_size, num_sentences, gen_file)  #TODO
                    # get_real_test_file(gen_file, gen_save_file, index_word_dict)
                    # get_real_test_file(gen_file, gen_text_file, index_word_dict)

                    # write summaries
                    scores = [metric.get_score() for metric in metrics]
                    metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                    sum_writer.add_summary(metrics_summary_str, epoch)

                    # msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                    msg = str(epoch)
                    metric_names = [metric.get_name() for metric in metrics]
                    for (name, score) in zip(metric_names, scores):
                        # msg += ', ' + name + ': %.4f' % score
                        msg += ', ' + '%.4f' % score
                    print(msg)
                    log.write(msg)
                    log.write('\n')

        generate_and_save(sess, config, x_fake, num_sentences, extra_subfolder="MLE_output")  #TODO

                # save_checkpoint(saver, sess, os.path.join(log_dir, 'pretrain_checkpoint'), epoch)
        log.flush()
        print('Start adversarial training...')
        all_bleu_metrics = None
        first = True

        progress = tqdm(range(nadv_steps-sess.run(global_step)))
        for _ in progress:
            niter = sess.run(global_step)

            t0 = time.time()

            # adversarial training
            reinforce_rewards, samples_for_rewards = None, None
            for _ in range(config['gsteps']):
                if '_pg' in config['g_architecture'] and '_pg' in config['d_architecture']:
                    # if config['rl_method'] == 1:
                    #     samples_for_rewards, reinforce_rewards, first, all_bleu_metrics = _get_rewards_01(config, oracle_loader, x_fake_for_rewards, eof_code, sess, first, all_bleu_metrics)
                    # if config['rl_method'] == 2:
                    #     samples_for_rewards, reinforce_rewards, first, all_bleu_metrics = _get_rewards_02(config, oracle_loader, x_fake_for_rewards, given_num, r_x, r_gen_x, r_gen_x_sample, eof_code, sess, first, all_bleu_metrics)
                    samples_for_rewards, reinforce_rewards, first, all_bleu_metrics = get_rewards(config, loader, x_fake,
                                                                                                      given_num, r_x,
                                                                                                      r_gen_x,
                                                                                                      r_gen_x_sample, sess)
                    _, g_loss_np, d_loss_np, loss_summary_str, reinforce_loss_str = sess.run([g_train_op, g_loss, d_loss, loss_summary_op, reinforce_loss], feed_dict={x_real: oracle_loader.random_batch(), rewards: reinforce_rewards, initial_samples_for_rewards: samples_for_rewards})
                else:
                    sess.run(g_train_op, feed_dict={x_real: oracle_loader.random_batch()})
            for _ in range(config['dsteps']):
                sess.run(d_train_op, feed_dict={x_real: oracle_loader.random_batch()})

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            # temperature
            # temp_var_np = get_fixed_temperature(temper, niter, nadv_steps, adapt)
            # sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

            if '_pg' in config['g_architecture'] and '_pg' in config['d_architecture']:
                # if config['rl_method'] == 1:
                #     samples_for_rewards, reinforce_rewards, first, all_bleu_metrics = _get_rewards_01(config, oracle_loader, x_fake_for_rewards, eof_code, sess, first, all_bleu_metrics)
                # elif config['rl_method'] == 2:
                #     samples_for_rewards, reinforce_rewards, first, all_bleu_metrics = _get_rewards_02(config, oracle_loader, x_fake_for_rewards, given_num, r_x, r_gen_x, r_gen_x_sample, eof_code, sess, first, all_bleu_metrics)
                # feed = {x_real: oracle_loader.random_batch(), rewards: reinforce_rewards, initial_samples_for_rewards: samples_for_rewards}
                pass
            else:
                feed = {x_real: oracle_loader.random_batch()}
                g_loss_np, d_loss_np, loss_summary_str, reinforce_loss_str = sess.run([g_loss, d_loss, loss_summary_op, reinforce_loss], feed_dict=feed)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)
            # progress.set_description('gbl_step: %d, g_loss: %4.4f, d_loss: %4.4f, REINF %4.4f, tempr: %4.4f' % (niter, g_loss_np, d_loss_np, reinforce_loss_str, temp_var_np))
            progress.set_description('gbl_step: %d, g_loss: %4.4f, d_loss: %4.4f, REINF %4.4f' % (niter, g_loss_np, d_loss_np, reinforce_loss_str))

            # Test
            if np.mod(niter, config['ntest']) == 0:
                # generate fake data and create batches
                # gen_save_file = os.path.join(sample_dir, 'adv_samples_{:05d}.txt'.format(niter))
                # samples_codes = generate_samples(sess, x_fake, batch_size, num_sentences, gen_file)
                # save_samples(samples_codes, os.path.join(sample_dir, 'generator_{:05d}.txt'.format(niter)))
                # get_real_test_file(gen_file, gen_save_file, index_word_dict)
                # get_real_test_file(gen_file, gen_text_file, index_word_dict)
                generate_and_save(sess, config, x_fake, num_sentences, extra_subfolder=f"adv_{niter}")  # TODO

                # write summaries
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, niter + config['npre_epochs'])

                # msg = 'adv_step: ' + str(niter)
                msg = str(niter)
                # metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    # msg += ', ' + name + ': %.4f' % score
                    msg += ', ' + '%.4f' % score
                print(msg)
                log.write(msg)
                log.write('\n')

            if np.mod(niter, config['checkpt_every']) == 0:
                save_checkpoint(saver_intermediate, sess, os.path.join(log_dir, 'checkpt_inter'), niter)

            if not config['no_excessive_debug']:
                save_checkpoint(saver, sess, os.path.join(log_dir, 'checkpt'), niter)

            log.flush()

        save_checkpoint(saver, sess, os.path.join(log_dir, 'checkpt'), niter)




# A function to get different GAN losses
def get_losses(d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr, gen_o, discriminator, config, rewards=None, initial_samples_for_rewards=None):
    batch_size = config['batch_size']
    gan_type = config['gan_type']
    seq_len = config['seq_len']
    vocab_size = config['vocab_size']
    RL_alpha = config['rl_alpha']

    if gan_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.ones_like(d_out_fake)
        ))

    elif gan_type == 'JS':  # the vanilla GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)

    elif gan_type == 'hinge':  # the hinge loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'tv':  # the total variation distance
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))

    elif gan_type == 'wgan-gp':  # WGAN-GP
        d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
        GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
        d_loss += GP

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'LS':  # LS-GAN
        d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))

    elif gan_type == 'RSGAN':  # relativistic standard GAN
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
        ))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
        ))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    #TODO
    reinforce_loss = tf.constant(0.0)
    if '_pg' in config['g_architecture'] and '_pg' in config['d_architecture']:
        reshaped_fake_one_hot = tf.reshape(x_fake_onehot_appr, [-1, vocab_size])
        rnn_outputs_for_reinforce = tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(initial_samples_for_rewards, [-1])), vocab_size, 1.0,
                       0.0) * tf.log(tf.clip_by_value(reshaped_fake_one_hot, 1e-20, 1.0)), 1)  # initial_samples_for_rewards[:, 1:]
        reinforce_loss = tf.reduce_mean(rnn_outputs_for_reinforce * tf.reshape(rewards, [-1]))  # reinforce_rewards[:, 1:]
        # reinforce_loss = tf.reduce_sum(rnn_outputs_for_reinforce * tf.reshape(rewards, [-1]))  # reinforce_rewards[:, 1:]
        reinforce_loss = - RL_alpha * reinforce_loss
        if config['rl_only'] == True:
            print("No gan objective in G, only policy gradients")
            g_loss = reinforce_loss
            d_loss = tf.get_variable("dummy_d_loss", initializer=0.0, trainable=False)
        else:
            g_loss += reinforce_loss

    log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)

    return log_pg, g_loss, d_loss, reinforce_loss


# A function to calculate the gradients and get training operations
def get_train_ops(config, g_pretrain_loss, g_loss, d_loss, log_pg, temperature, global_step):
    optimizer_name = config['optimizer']
    nadv_steps = config['nadv_steps']
    d_lr = config['d_lr']
    gpre_lr = config['gpre_lr']
    gadv_lr = config['gadv_lr']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    grad_clip = 5.0 if "rmc_vanilla" in config['g_architecture'] else 10.0  # keep the same with the previous setting

    # generator pre-training
    pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999)
    pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars), grad_clip)  # gradient clipping
    g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    # decide if using the weight decaying
    if config['decay']:
        d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        gadv_lr = tf.train.exponential_decay(gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

    # Adam optimizer
    if optimizer_name == 'adam':
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        g_optimizer = tf.train.AdamOptimizer(gadv_lr, beta1=0.9, beta2=0.999)
        temp_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.9, beta2=0.999)

    # RMSProp optimizer
    elif optimizer_name == 'rmsprop':
        d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
        temp_optimizer = tf.train.RMSPropOptimizer(1e-2)

    else:
        raise NotImplementedError

    # gradient clipping
    g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), grad_clip)
    g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))

    print('len of g_grads without None: {}'.format(len([i for i in g_grads if i is not None])))
    print('len of g_grads: {}'.format(len(g_grads)))

    # gradient clipping
    if "dummy_d_loss" in d_loss.name:
        d_train_op = tf.constant(0.0)
    else:
        d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars), grad_clip)
        d_train_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))


    return g_pretrain_op, g_train_op, d_train_op


# A function to get various evaluation metrics
def get_metrics(config, oracle_loader, test_file, gen_file, g_pretrain_loss, x_real, sess):
    # set up evaluation metric
    metrics = list()
    if config['nll_gen']:
        nll_gen = Nll(oracle_loader, g_pretrain_loss, x_real, sess, name='nll_gen')
        metrics.append(nll_gen)
    if config['doc_embsim']:
        doc_embsim = DocEmbSim(test_file, gen_file, config['vocab_size'], name='doc_embsim')
        metrics.append(doc_embsim)
    if config['bleu']:
        for i in range(2, 6):
            bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name='bleu' + str(i))
            metrics.append(bleu)
    if config['selfbleu']:
        for i in range(2, 6):
            selfbleu = SelfBleu(test_text=gen_file, gram=i, name='selfbleu' + str(i))
            metrics.append(selfbleu)

    return metrics


# A function to get the summary for each metric
def get_metric_summary_op(config):
    metrics_pl = []
    metrics_sum = []

    if config['nll_gen']:
        nll_gen = tf.placeholder(tf.float32)
        metrics_pl.append(nll_gen)
        metrics_sum.append(tf.summary.scalar('metrics/nll_gen', nll_gen))

    if config['doc_embsim']:
        doc_embsim = tf.placeholder(tf.float32)
        metrics_pl.append(doc_embsim)
        metrics_sum.append(tf.summary.scalar('metrics/doc_embsim', doc_embsim))

    if config['bleu']:
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='bleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/bleu{}'.format(i), temp_pl))

    if config['selfbleu']:
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='selfbleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/selfbleu{}'.format(i), temp_pl))

    mcgrew = tf.placeholder(tf.float32, "mcgrew_scores")
    metrics_pl.append(mcgrew)
    metrics_sum.append(tf.summary.scalar('scores/MCgrew_Score', mcgrew))

    metric_summary_op = tf.summary.merge(metrics_sum)
    return metrics_pl, metric_summary_op

# A function to set up different temperature control policies
# def get_fixed_temperature(temper, i, N, adapt):
#     if adapt == 'no':
#         temper_var_np = temper  # no increase
#     elif adapt == 'lin':
#         temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
#     elif adapt == 'exp':
#         temper_var_np = temper ** (i / N)  # exponential increase
#     elif adapt == 'log':
#         temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
#     elif adapt == 'sigmoid':
#         temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
#     elif adapt == 'quad':
#         temper_var_np = (temper - 1) / (N - 1)**2 * i ** 2 + 1
#     elif adapt == 'sqrt':
#         temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
#     else:
#         raise Exception("Unknown adapt type!")
#
#     return temper_var_np
#
# def _get_rewards_02(config, data_loader, x_fake_for_rewards, given_num, r_x, r_gen_x, r_gen_x_sample, eof_code, sess, first, all_bleu_metrics):
#     # print("Start computing rewards ...")
#     batch_size = config['batch_size']
#     gan_type = config['gan_type']
#     seq_len = config['seq_len']
#     vocab_size = config['vocab_size']
#     rl_bleu_ref_count = data_loader.num_batch * batch_size  # all of training set
#     # rl_n_grams = 4
#     rl_mc_samples = config['mc_samples']
#     gamma_discount = 0.5
#
#     # rewards = np.zeros((batch_size, seq_len), np.float32)
#
#     if first == True:
#         train_refs = data_loader.get_as_lol_no_padding()
#
#         # train_refs = data_loader.random_some(rl_bleu_ref_count, seq_len + 1)
#         bleu_metric_2 = Bleu.from_references_indices(2, train_refs)
#
#         # train_refs = data_loader.random_some(rl_bleu_ref_count, seq_len + 1)
#         bleu_metric_3 = Bleu.from_references_indices(3, train_refs)
#
#         # train_refs = data_loader.random_some(rl_bleu_ref_count, seq_len + 1)
#         bleu_metric_4 = Bleu.from_references_indices(4, train_refs)
#
#         # train_refs = data_loader.random_some(rl_bleu_ref_count, seq_len + 1)
#         bleu_metric_5 = Bleu.from_references_indices(5, train_refs)
#
#         all_bleu_metrics = [bleu_metric_2, bleu_metric_3, bleu_metric_4, bleu_metric_5]
#
#         first = False
#
#     rewards = list()
#     samples_for_rewards = sess.run(x_fake_for_rewards)
#
#     for i in range(rl_mc_samples):
#         for given_num_i in range(1, seq_len):
#             feed = {r_x: samples_for_rewards, given_num: given_num_i}
#             roll_out_samples = sess.run(r_gen_x, feed)
#             # feed = {discriminator.input_x: samples}
#             # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
#             ypred = _compute_rl_rewards_02(roll_out_samples, all_bleu_metrics, gamma_discount, eof_code)
#             # ypred = np.array([item[1] for item in ypred_for_auc])
#             if i == 0:
#                 rewards.append(ypred)
#             else:
#                 rewards[given_num_i - 1] += ypred
#
#         # the last token reward
#         # feed = {discriminator.input_x: input_x}
#         # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
#         # ypred = np.array([item[1] for item in ypred_for_auc])
#         ypred = _compute_rl_rewards_02(samples_for_rewards, all_bleu_metrics, gamma_discount, eof_code)
#         if i == 0:
#             rewards.append(ypred)
#         else:
#             rewards[(len(samples_for_rewards[0]) - 1)] += ypred
#
#     # for _ in range(rl_mc_samples):
#     #     # samples_for_rewards, _ = self.generator.generate_from_noise(self.sess, batch_size, self.current_tau, Config.args.BATCH_SIZE)
#     #     samples_for_rewards = sess.run(x_fake_for_rewards)
#     #     for b in range(len(samples_for_rewards)):
#     #         rewards[b, :] = rewards[b, :] + _compute_rl_rewards(samples_for_rewards[b], all_bleu_metrics, gamma_discount)
#     # rewards = rewards / (1.0 * rl_mc_samples)
#     # return samples_for_rewards, rewards
#
#     reward_res = np.transpose(np.array(rewards)) / (1.0 * rl_mc_samples)  # batch_size x seq_length
#     if config['pg_baseline']:
#         reward_res -= config['pg_baseline_val']  # 2.0 for emnlp
#     # print("Rewards computed.")
#     return samples_for_rewards, reward_res, first, all_bleu_metrics
#
# def _compute_rl_rewards_02(gen_seq, bleu_metrics, gamma_discount, eof_code):
#     """
#     Expects a gen_seq of shape (T,) tokens
#     :return: rewards values for given gen_seq
#     """
#     # gd = np.reshape(gen_seq, [1, gen_seq.shape[0]])
#
#     # T = seq_returns.shape[1]
#     # reward_T_plus_1 =  0.0  # get_all_bleus_new(n_grams, gd[0, :], refs)
#     # seq_returns[0, T - 1] = 0.0
#
#     # for time_step in range(T - 2, - 1, -1):
#     gen_seq_list = samples_no_padding(gen_seq, eof_code)
#     reward_t_plus_1_2g = bleu_metrics[0].get_all_sentence_bleus_fast(gen_seq_list, smoothing_method=1)
#     reward_t_plus_1_3g = bleu_metrics[1].get_all_sentence_bleus_fast(gen_seq_list, smoothing_method=1)
#     reward_t_plus_1_4g = bleu_metrics[2].get_all_sentence_bleus_fast(gen_seq_list, smoothing_method=1)
#     reward_t_plus_1_5g = bleu_metrics[3].get_all_sentence_bleus_fast(gen_seq_list, smoothing_method=1)
#
#     # reward_t_plus_1_2g = bleu_metrics[0].get_bleu_parallel(gen_seq_list)
#     # # print("done 1/4 of rewards ...")
#     # reward_t_plus_1_3g = bleu_metrics[1].get_bleu_parallel(gen_seq_list)
#     # # print("done 2/4 of rewards ...")
#     # reward_t_plus_1_4g = bleu_metrics[2].get_bleu_parallel(gen_seq_list)
#     # # print("done 3/4 of rewards ...")
#     # reward_t_plus_1_5g = bleu_metrics[3].get_bleu_parallel(gen_seq_list)
#     # # print("done 4/4 of rewards ...")
#
#     reward_t_plus_1 = np.array(reward_t_plus_1_2g) + np.array(reward_t_plus_1_3g) + np.array(reward_t_plus_1_4g) + np.array(reward_t_plus_1_5g)
#     seq_returns = reward_t_plus_1  # + gamma_discount * seq_returns[0, time_step+1]
#
#     return seq_returns


# def _save_samples(self, generated_list_of_seq, output_folder):
#     for g in range(len(generated_list_of_seq)):
#         gen_seq = generated_list_of_seq[g]
#         df = self._prepare_for_saving(gen_seq)
#         df.to_csv(output_folder + r"/ace_generated_run_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S%f") + ".csv", index=False)
#         time.sleep(0.1)

def generate_and_save(sess, config, gen_x, generated_num, extra_subfolder=""):
    """
    Generate and save random sentences
    """
    # if not os.path.exists(os.path.join(config['sample_dir'], extra_subfolder, "real_x1")):
    #     os.makedirs(os.path.join(config['sample_dir'], extra_subfolder, "real_x1"))
    if not os.path.exists(os.path.join(config['sample_dir'], extra_subfolder)):
        os.makedirs(os.path.join(config['sample_dir'], extra_subfolder))

    input_condition = None
    # if Config.args.cond:
    #     all_data = np.concatenate((self.train, self.test), 0)
    #     first_inputs = all_data[:, 0]
    #     input_condition = all_data[:, :, int(self.input_size / 2):]
    #     n_inputs = all_data.shape[0]
    # else:
    # first_inputs = self.test[:, 0]
    # n_inputs = first_inputs.shape[0]

    generated_samples = list()
    for _ in range(int(generated_num / config['batch_size'])):
        generated_samples.extend(sess.run(gen_x))

    # if Config.args.N_SENTENCES < Config.args.BATCH_SIZE:
    #     n_samples_to_generate = Config.args.BATCH_SIZE
    # else:
    #     n_samples_to_generate = Config.args.N_SENTENCES - (Config.args.N_SENTENCES % Config.args.BATCH_SIZE)

    # generated_list_of_seq_real = list()
    # generated_list_of_seq_noise = list()

    # for i_sample_iter in range(n_samples_to_generate // n_inputs + 1):
    #
    #     from_real = self.generator.generate_from_real_x1(self.sess, first_inputs, n_inputs, Config.args.BATCH_SIZE,
    #                                                      use_argmax=None, input_condition=input_condition)
    #     generated_list_of_seq_real += from_real
    #
    #     if Config.args.use_vae:
    #         from_noise, _ = self.generator.generate_from_noise(self.sess, n_inputs, Config.args.BATCH_SIZE,
    #                                                            use_argmax=None, input_condition=input_condition)
    #         generated_list_of_seq_noise += from_noise
    #
    # generated_list_of_seq_real = generated_list_of_seq_real[
    #                              :n_samples_to_generate]  # Take only the needed n_samples_to_generate sentences
    # print("generated_list_of_seq_real_x1 = ", len(generated_list_of_seq_real))
    # self._save_samples(generated_list_of_seq_real,
    #                    os.path.join(config['sample_dir'], extra_subfolder, "real_x1"))

    # generated_list_of_seq_noise = generated_list_of_seq_noise[:n_samples_to_generate]
    print("generated_list_of_seq_noise = ", len(generated_samples))
    _save_samples(generated_samples, os.path.join(config['sample_dir'], extra_subfolder))

    sys.stdout.flush()


def _save_samples(generated_list_of_seq, output_folder):
    for g in range(len(generated_list_of_seq)):
        gen_seq = generated_list_of_seq[g]
        df = _prepare_for_saving(gen_seq)
        df.to_csv(output_folder + r"/ace_generated_run_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S%f") + ".csv",
                  index=False)
        time.sleep(0.1)


def _prepare_for_saving(gen_seq, output_size):
    gd = _descale_data(gen_seq, output_size)
    cols = ["timestep", "x", "y", "z", "psi", "theta", "phi", "v", "gload", "mcgrew_score"]
    n_out_columns = int(output_size / 2)
    gd = _append_mcgrew_scores(gd, n_out_columns)
    sim_time_step = 1.0
    time_step_column = [float(i) * sim_time_step for i in range(gen_seq.shape[0])]
    time_step_column = np.reshape(time_step_column, (-1, 1))
    viper_df = pd.DataFrame(np.concatenate((time_step_column, gd[:, :((int(output_size / 2))+1)]), axis=1), columns=cols)
    cobra_df = pd.DataFrame(np.concatenate((time_step_column, gd[:, ((int(output_size / 2))+1):]), axis=1), columns=cols)
    viper_df.insert(1, "callsign", "viper")
    cobra_df.insert(1, "callsign", "cobra")

    both_df = pd.concat([viper_df, cobra_df])
    return both_df

def _append_mcgrew_scores(descaled_gen_seq, n_out_columns):
    """
    :return: generated sequence after appending the Mcgrew score with shape [time_steps, out_columns + 1]
    """
    viper_seq = descaled_gen_seq[:, :n_out_columns]
    viper_seq_rewards = np.zeros((viper_seq.shape[0], n_out_columns + 1))
    viper_seq_rewards[:, :-1] = viper_seq

    cobra_seq = descaled_gen_seq[:, n_out_columns:]
    cobra_seq_rewards = np.zeros((cobra_seq.shape[0], n_out_columns + 1))
    cobra_seq_rewards[:, :-1] = cobra_seq

    T = descaled_gen_seq.shape[0]

    for time_step in range(T):
        viper_score = utils_ace0.contact_assessment(utils_ace0.build_state(viper_seq[time_step, :]),
                                        utils_ace0.build_state(cobra_seq[time_step, :]))
        cobra_score = utils_ace0.contact_assessment(utils_ace0.build_state(cobra_seq[time_step, :]),
                                        utils_ace0.build_state(viper_seq[time_step, :]))
        viper_seq_rewards[time_step, -1] = viper_score
        cobra_seq_rewards[time_step, -1] = cobra_score

    return np.concatenate((viper_seq_rewards, cobra_seq_rewards), axis=1)

def get_rewards(config, data_loader, x_fake_for_rewards, given_num, r_x, r_gen_x, r_gen_x_sample, sess):
    gamma_discount = 0.9
    batch_size = config['batch_size']
    output_size = config['gen_emb_dim']
    rewards = np.zeros((batch_size, config['seq_len']), np.float32)
    n_out_columns = int(config['vocab_size'] / 2)
    mc_samples = config['mc_samples']
    for i in range(mc_samples): #TODO[fixme] should this be 1 not many ?
        input_condition = 0.0  # Dummy
        # if Config.args.cond:
        #     data_to_use = self.train
        #     random_minibatches = self.__get_minibatches_indecies(data_to_use.shape[0], shuffle=True)
        #     # first_inputs = data_to_use[random_minibatches [0], 0]
        #     input_condition = data_to_use[random_minibatches[0], :, int(self.input_size / 2):]

        # samples_for_rewards = sess.run(x_fake_for_rewards)
        generated_list_of_seq = sess.run(x_fake_for_rewards)
        # feed = {r_x: samples_for_rewards, given_num: given_num_i}
        # roll_out_samples = sess.run(r_gen_x, feed)

        # generated_list_of_seq, _ = generator.generate_from_noise(sess, batch_size, batch_size, use_argmax=None, input_condition=input_condition)

        for b in range(len(generated_list_of_seq)):
            gen_seq = generated_list_of_seq[b]
            rewards[b, :] = rewards[b, :] + _compute_rl_rewards(gen_seq, n_out_columns, gamma_discount, output_size, data_loader)
    rewards = rewards / (1.0 * mc_samples)
    return rewards

# def _get_rewards_02(config, data_loader, x_fake_for_rewards, given_num, r_x, r_gen_x, r_gen_x_sample, eof_code, sess, first, all_bleu_metrics):
#     # print("Start computing rewards ...")
#     batch_size = config['batch_size']
#     gan_type = config['gan_type']
#     seq_len = config['seq_len']
#     vocab_size = config['vocab_size']
#     rl_bleu_ref_count = data_loader.num_batch * batch_size  # all of training set
#     # rl_n_grams = 4
#     rl_mc_samples = config['mc_samples']
#     gamma_discount = 0.5
#
#     rewards = list()
#     samples_for_rewards = sess.run(x_fake_for_rewards)
#
#     for i in range(rl_mc_samples):
#         for given_num_i in range(1, seq_len):
#             feed = {r_x: samples_for_rewards, given_num: given_num_i}
#             roll_out_samples = sess.run(r_gen_x, feed)
#             # feed = {discriminator.input_x: samples}
#             # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
#             ypred = _compute_rl_rewards_02(roll_out_samples, all_bleu_metrics, gamma_discount, eof_code)
#             # ypred = np.array([item[1] for item in ypred_for_auc])
#             if i == 0:
#                 rewards.append(ypred)
#             else:
#                 rewards[given_num_i - 1] += ypred
#
#         # the last token reward
#         # feed = {discriminator.input_x: input_x}
#         # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
#         # ypred = np.array([item[1] for item in ypred_for_auc])
#         ypred = _compute_rl_rewards_02(samples_for_rewards, all_bleu_metrics, gamma_discount, eof_code)
#         if i == 0:
#             rewards.append(ypred)
#         else:
#             rewards[(len(samples_for_rewards[0]) - 1)] += ypred
#
#     # for _ in range(rl_mc_samples):
#     #     # samples_for_rewards, _ = self.generator.generate_from_noise(self.sess, batch_size, self.current_tau, Config.args.BATCH_SIZE)
#     #     samples_for_rewards = sess.run(x_fake_for_rewards)
#     #     for b in range(len(samples_for_rewards)):
#     #         rewards[b, :] = rewards[b, :] + _compute_rl_rewards(samples_for_rewards[b], all_bleu_metrics, gamma_discount)
#     # rewards = rewards / (1.0 * rl_mc_samples)
#     # return samples_for_rewards, rewards
#
#     reward_res = np.transpose(np.array(rewards)) / (1.0 * rl_mc_samples)  # batch_size x seq_length
#     if config['pg_baseline']:
#         reward_res -= config['pg_baseline_val']  # 2.0 for emnlp
#     # print("Rewards computed.")
#     return samples_for_rewards, reward_res, first, all_bleu_metrics

def _compute_rl_rewards(gen_seq, n_out_columns, gamma_discount, output_size, loader):
    """
    :return: numpy array of rewards with shape [1, time_steps]
    """
    gd = _descale_data(gen_seq, output_size, loader)

    viper_seq = gd[:, :n_out_columns]
    viper_seq_returns = np.zeros((viper_seq.shape[0], n_out_columns + 1))
    viper_seq_returns[:, :-1] = viper_seq

    cobra_seq = gd[:, n_out_columns:]
    T = gen_seq.shape[0]

    viper_reward_T_plus_1 = 0.0
    viper_seq_returns[T - 1, -1] = 0.0

    for time_step in range(T - 2, -1, -1):
        viper_reward_t_plus_1 = utils_ace0.contact_assessment(utils_ace0.build_state(viper_seq[time_step+1, :]), utils_ace0.build_state(cobra_seq[time_step+1, :]))
        # cobra_score = utils_ace0.contact_assessment(utils_ace0.build_state(cobra_seq[time_step+1, :]),
        #                                 utils_ace0.build_state(viper_seq[time_step+1, :]))
        viper_seq_returns[time_step, -1] = viper_reward_t_plus_1 + gamma_discount * viper_seq_returns[time_step+1, -1]

    return np.reshape(viper_seq_returns[:, -1], [-1, gen_seq.shape[0]])

def _descale_data(gen_seq, output_size, loader):
    gen_descaled = None
    # elif Config.args.normz == 'featr_only':
    my_gen = np.reshape(gen_seq, [-1, output_size])
    gen_descaled = loader.scalers[0].inverse_transform(my_gen)
    gen_descaled = np.reshape(gen_descaled, (gen_seq.shape[0], gen_seq.shape[1]))

    return gen_descaled


