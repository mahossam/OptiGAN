import tensorflow as tf
from models import rmc_vanilla, rmc_att, rmc_vdcnn, lstm_vanilla, lstm_vanilla_pg, lstm_vanilla_new_init, lstm_vanilla_pg_new_init, lstm_vanilla_pg_mle_gan_new_init, rmc_vanilla_pg, lstm_vanilla_new_init_contin

generator_dict = {
    'lstm_vanilla': lstm_vanilla.generator,
    'lstm_vanilla_pg': lstm_vanilla_pg.generator,
    'lstm_vanilla_new_init': lstm_vanilla_new_init.generator,
    'lstm_vanilla_new_init_contin': lstm_vanilla_new_init_contin.generator,
    'lstm_vanilla_pg_new_init': lstm_vanilla_pg_new_init.generator,
    'lstm_vanilla_pg_mle_gan_new_init': lstm_vanilla_pg_mle_gan_new_init.generator,
    'rmc_vanilla': rmc_vanilla.generator,
    'rmc_vanilla_pg': rmc_vanilla_pg.generator,
    'rmc_att': rmc_att.generator,
    'rmc_vdcnn': rmc_vdcnn.generator
}

discriminator_dict = {
    'lstm_vanilla': lstm_vanilla.discriminator,
    'lstm_vanilla_pg': lstm_vanilla_pg.discriminator,
    'lstm_vanilla_new_init': lstm_vanilla_new_init.discriminator,
    'lstm_vanilla_new_init_contin': lstm_vanilla_new_init_contin.discriminator,
    'lstm_vanilla_pg_new_init': lstm_vanilla_pg_new_init.discriminator,
    'lstm_vanilla_pg_mle_gan_new_init': lstm_vanilla_pg_mle_gan_new_init.discriminator,
    'rmc_vanilla': rmc_vanilla.discriminator,
    'rmc_vanilla_pg': rmc_vanilla_pg.discriminator,
    'rmc_att': rmc_att.discriminator,
    'rmc_vdcnn': rmc_vdcnn.discriminator
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)