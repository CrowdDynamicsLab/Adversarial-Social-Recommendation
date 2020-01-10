import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import codecs
import time
from scipy import sparse
import pandas as pd
from eval_functions import ndcg_binary_at_k_batch, recall_at_k_batch, overlap_at_k_batch
from data_process import load_dimension, load_train_data, load_vad_tr_te_data, load_te_tr_te_data, load_rating_matrix, \
    load_user_data, \
    load_adj_matrix
from generator import generator_vaecf as generator
from discriminator import discriminator
from sample import sample_from_generator_new
from VAE import MultiVAE
from layers import CustomDense
import flags
from flags import FLAGS


# The dataset directory
DATA_DIR = FLAGS.data_dir


# Number of social features
N_GCN_FEATURES = FLAGS.gcn_feature_count

# Percentage of users whose true and fake pairs are considered per minibatch
# Typically 1-5% (use less for larger datasets)
BATCH_SIZE_PERCENTAGE = FLAGS.batch_size_percentage


# Number of fake pairs per user (as a percentage of all users), 1-5%
# 0.05 implies 5% users are randomly sampled to create fake pairs per user
USER_SAMPLE_PERCENTAGE = FLAGS.user_sample_percentage



# Each global epoch involves a resampling of fake and true pairs
# Each global epoch also includes sub epochs for generator, discrim and pair weight (memory) module
NUM_GLOBAL_EPOCH = FLAGS.global_epochs
NUM_GEN_SUB_EPOCHS = FLAGS.gen_sub_epochs
NUM_DISC_SUB_EPOCHS = FLAGS.disc_sub_epochs
NUM_MEMORY_SUB_EPOCHS = FLAGS.mem_sub_epochs

# Learning rates - Important to tune
DISCRIMINATOR_LEARNING_RATE = FLAGS.d_learning_rate
GENERATOR_LEARNING_RATE = FLAGS.g_learning_rate
MEMORY_LEARNING_RATE = FLAGS.m_learning_rate


# Can change if required
dim_placeholder = 0
DECODER_DIMS = [200, 600, dim_placeholder]
ENCODER_DIMS = [dim_placeholder, 600, 200]

# The gan lambda paramter introduced in the paper
GANLAMBDA = FLAGS.ganlambda


# Number of keys in memory for pair weighting
NUM_MEMORY_BLOCKS = FLAGS.memory_blocks

# Discriminator Layer sizes
DISCRIMINATOR_HIDDEN_1 = FLAGS.d_hidden_1
DISCRIMINATOR_HIDDEN_2 = FLAGS.d_hidden_2

# Balance parameter for discriminator training
POSITIVE_NEGATIVE_MU = FLAGS.mu

# Dropout 
drop_percent = FLAGS.dropout_rate






def get_mask(idxlist, st_idx, end_idx, num_user, offset=0):
    batch_size = end_idx - st_idx
    mask = np.ones((batch_size, num_user))
    for i in range(batch_size):
        mask[i, offset + idxlist[st_idx + i]] = 0
    return mask


#####################################   MEMORY MODULE   #############################################################################################

def get_pair_weights(user_1_id, user_2_id, all_user_emb, num_memory_blocks, d):
    
    # Contextual Pair-Weighting mechanism

    keys = tf.Variable(tf.truncated_normal([num_memory_blocks, d], dtype=tf.float32, stddev=0.1))
    user_1_z = tf.nn.embedding_lookup(all_user_emb, user_1_id)
    user_2_z = tf.nn.embedding_lookup(all_user_emb, user_2_id)

    user_1_z_transformed = tf.matmul(user_1_z, tf.transpose(keys))
    user_2_z_transformed = tf.matmul(user_2_z, tf.transpose(keys))

    hadamard_prod = tf.multiply(user_1_z_transformed, user_2_z_transformed)

    # The hadamard_prod represents the similarities of users as a vector (each dimension is the similarity measured along one key)
    # There are num_memory_block keys (i.e., each block is a key) 

    # There are two options for the output - Feed-forward layer to transform the similarities, or a simple dot product

    ff_layer = CustomDense(num_memory_blocks, 1, name = "memory_layer", dropout=FLAGS.use_dropout, dropout_rate=FLAGS.dropout_rate)
    dot_prod_weight = tf.Variable(tf.truncated_normal([num_memory_blocks, 1], dtype=tf.float32, stddev=0.1))

    # hadamard_prod_scaled = tf.matmul(hadamard_prod, prob)
    hadamard_prod_scaled = ff_layer.__call__(hadamard_prod)

    # Finally scale the output to a suitable range
    pair_weights = tf.add(-1.0, 2.0*tf.nn.sigmoid(hadamard_prod_scaled))

    # Return memory module parameters
    memory_params = [keys, dot_prod_weight, ff_layer.vars['weights'], ff_layer.vars['bias']]

    return pair_weights, memory_params



#####################################   LOADING AND INITIALIZATION   ################################################################################

n_items, n_users = load_dimension(DATA_DIR)
train_data = load_train_data(DATA_DIR, n_items, n_users, drop_percent)
uid_start_idx = 0
vad_data_tr, vad_data_te = load_vad_tr_te_data(DATA_DIR, n_items, n_users)
te_data_tr, te_data_te = load_te_tr_te_data(DATA_DIR, n_items, n_users)
rating_matrix = load_rating_matrix(DATA_DIR, n_items, n_users)
user_matrix = load_user_data(DATA_DIR, n_users)  # numpy array, [n_users,n_users]
print("Loaded %d items and %d users" % (n_items, n_users))
adj_matrix = [load_adj_matrix(DATA_DIR, drop_percent)]

train_data_arr = train_data.toarray().astype('float32')
vad_data_arr = vad_data_tr.toarray().astype('float32')
te_data_arr = te_data_tr.toarray().astype('float32')

rating_train_matrix = np.vstack((np.vstack((train_data_arr, vad_data_arr)), te_data_arr))


N_vad = vad_data_tr.shape[0]
N = train_data.shape[0]

USER_SAMPLE_NUM = int(USER_SAMPLE_PERCENTAGE * N)
BATCH_SIZE = int(BATCH_SIZE_PERCENTAGE * N)
GANLAMBDA *= USER_SAMPLE_NUM*BATCH_SIZE/DECODER_DIMS[0]

idxlist = range(N)
idxlist_vad = range(N_vad)

DECODER_DIMS[2] = n_items
ENCODER_DIMS[0] = n_items


print('Number of Users: ', N)

batches_per_epoch = int(np.ceil(float(N) / BATCH_SIZE))

print('Batches Per Epoch: ', batches_per_epoch)

global_step = tf.Variable(0, name="global_step", trainable=False)

tf.reset_default_graph()

best_recall_10 = 0.0
best_recall_20 = 0.0
best_recall_50 = 0.0


###############################################################     GENERATOR     ###############################################################


generator_network, item_distribution_out, g_vae_loss, g_params, total_anneal_steps, anneal_cap, user_emb = generator(
    n_items, n_users, rating_train_matrix, DECODER_DIMS, ENCODER_DIMS)

generated_tags = tf.placeholder(tf.float32, [None, n_users], name="generated_tags")


###############################################################     DISCRIMINATOR     ###############################################################


y_data, y_generated, x_generated_1_id, x_true_1_id, x_generated_2_id, x_true_2_id, placeholders, x_gen_1, x_gen_2, emb_matrix = discriminator(
    n_users, N_GCN_FEATURES, DISCRIMINATOR_HIDDEN_1, DISCRIMINATOR_HIDDEN_2)

zero = tf.constant(0, dtype=tf.float32)

all_user_emb = tf.placeholder(dtype=tf.float32, shape=[n_users, DECODER_DIMS[0]])


###############################################################     MEMORY_MODULE     ###############################################################


pair_weights, memory_module_params = get_pair_weights(x_generated_1_id, x_generated_2_id,
                                               all_user_emb, NUM_MEMORY_BLOCKS, DECODER_DIMS[0])


###############################################################                       ###############################################################



# Compute the similarities of a specific batch of users against all users
similarities_batch_all = tf.matmul(user_emb, tf.transpose(all_user_emb))
similarities_batch_all_sigmoid = tf.maximum(10**-5, tf.nn.sigmoid(similarities_batch_all))




# Loss Functions

d_loss = - tf.reduce_sum(tf.log(y_data)) - POSITIVE_NEGATIVE_MU * tf.reduce_sum(tf.log(1 - y_generated))
d_loss_mean = tf.reduce_mean(d_loss)


sampled_generator_out = tf.multiply(similarities_batch_all_sigmoid, generated_tags)
sampled_generator_out = tf.reshape(sampled_generator_out, [-1])
sampled_generator_out_non_zero = tf.gather_nd(sampled_generator_out,
                                              tf.where(tf.not_equal(sampled_generator_out, zero)))
sampled_generator_out_non_zero = tf.reshape(sampled_generator_out_non_zero, [-1])
sampled_generator_out_non_zero = tf.squeeze(sampled_generator_out_non_zero)
y_generated = tf.reshape(y_generated, [-1])
pair_weights = tf.reshape(pair_weights, [-1])
sampled_cnt = tf.placeholder_with_default(1., shape=None)
gen_lambda = tf.placeholder_with_default(1.0, shape=None)
gan_loss = - (1.0 * gen_lambda / sampled_cnt) * tf.reduce_sum(
    tf.multiply(pair_weights, tf.multiply(sampled_generator_out_non_zero, y_generated)))
g_loss = g_vae_loss + gan_loss
g_loss_mean = tf.reduce_mean(g_loss)


# Optimizers

d_optimizer = tf.train.AdamOptimizer(DISCRIMINATOR_LEARNING_RATE)
g_optimizer = tf.train.AdamOptimizer(GENERATOR_LEARNING_RATE)
memory_optimizer = tf.train.AdamOptimizer(MEMORY_LEARNING_RATE)

# Discriminator and generator loss

# Discriminator does not need param list since the loss does not depend on anything outside it

d_trainer = d_optimizer.minimize(d_loss)
g_trainer = g_optimizer.minimize(g_loss, var_list=g_params)
memory_trainer = memory_optimizer.minimize(gan_loss, var_list=memory_module_params)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)


# The curr_gen_lambda placeholder is used to allow for annealing - a process where the GANLAMBDA value may be changed over iterations

curr_gen_lamda = GANLAMBDA
update_count = 0.0
curr_all_user_emb = np.random.randn(n_users, DECODER_DIMS[0])
print('starting train')




###############################################################     MAIN TRAINING LOOP     ###############################################################




for i in range(NUM_GLOBAL_EPOCH):

    # 1. Sample a new set of fake, true pairs at the start of each global epoch
    # 2. Update Generator, Discriminator and Memory modules with the sampled pairs


    batch_total_sampled_tags = []
    batch_curr_x_generated_1 = []
    batch_curr_x_generated_2 = []
    batch_curr_x_true_1 = []
    batch_curr_x_true_2 = []
    batch_X = []
    batch_st_idx = []
    batch_total_sampled_cnt = []

    user_err_cnt = 0

    for bnum, st_idx in enumerate(range(0, N, BATCH_SIZE)):
        end_idx = min(st_idx + BATCH_SIZE, N)
        X = train_data[idxlist[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        curr_generator_out, curr_batch_emb = sess.run([similarities_batch_all_sigmoid, user_emb],
                                                      feed_dict={generator_network.input_ph: X,
                                                                 # batch_user_ids: idxlist[st_idx:end_idx],
                                                                 all_user_emb: curr_all_user_emb})
        curr_all_user_emb[idxlist[st_idx:end_idx]] = curr_batch_emb
        curr_x_true_1 = []
        curr_x_true_2 = []

        curr_x_generated_1 = []
        curr_x_generated_2 = []

        total_sampled_cnt = 0
        total_sampled_tags = []

        # generate true and fake pairs
        for ii, user_idx in enumerate(idxlist[st_idx:end_idx]):
       
            curr_sampled_tags_bin, curr_sampled_users = sample_from_generator_new(range(n_users),
                                                                                  np.asarray(curr_generator_out)[
                                                                                      ii], USER_SAMPLE_NUM, n_users)

            curr_cnt = USER_SAMPLE_NUM
            curr_sampled_users.sort()

            # get fake pairs here
            curr_x_generated_1 += [user_idx] * USER_SAMPLE_NUM
            curr_x_generated_2 += list(curr_sampled_users)

            # get true pairs here
            curr_x_true_1 += [user_idx] * USER_SAMPLE_NUM
            _, sampled_true = sample_from_generator_new(range(n_users), user_matrix[user_idx], USER_SAMPLE_NUM,
                                                        n_users)
            curr_x_true_2 += list(sampled_true)

            # possibly shuffle each pair so that first pair is not always input user

            total_sampled_tags.append(curr_sampled_tags_bin)
            total_sampled_cnt += curr_cnt

        if curr_x_generated_1 == []:
            continue

        total_sampled_tags = np.asarray(total_sampled_tags)
        curr_x_generated_1 = np.asarray(curr_x_generated_1)
        curr_x_generated_2 = np.asarray(curr_x_generated_2)
        curr_x_true_1 = np.asarray(curr_x_true_1)
        curr_x_true_2 = np.asarray(curr_x_true_2)

        batch_total_sampled_tags.append(total_sampled_tags)
        batch_curr_x_generated_1.append(curr_x_generated_1)
        batch_curr_x_generated_2.append(curr_x_generated_2)
        batch_curr_x_true_1.append(curr_x_true_1)
        batch_curr_x_true_2.append(curr_x_true_2)
        batch_X.append(X)
        batch_st_idx.append(st_idx)
        batch_total_sampled_cnt.append(total_sampled_cnt)

    batch_total_sampled_tags = np.asarray(batch_total_sampled_tags)
    batch_curr_x_generated_1 = np.asarray(batch_curr_x_generated_1)
    batch_curr_x_generated_2 = np.asarray(batch_curr_x_generated_2)
    batch_curr_x_true_1 = np.asarray(batch_curr_x_true_1)
    batch_curr_x_true_2 = np.asarray(batch_curr_x_true_2)
    batch_X = np.asarray(batch_X)
    batch_st_idx = np.asarray(batch_st_idx)
    batch_total_sampled_cnt = np.asarray(batch_total_sampled_cnt)

    print("global-epoch:", i, "Data Creation Finished", "user_err_cnt:", user_err_cnt)

    # print(batch_total_sampled_cnt.tolist())

    indices = np.arange(batch_total_sampled_tags.shape[0])
    np.random.shuffle(indices)


    ########### GENERATOR SUB EPOCHS ###########


    for j_gen in range(NUM_GEN_SUB_EPOCHS):

        for gen_batch_idx in indices:
            X = batch_X[gen_batch_idx]
            st_idx = batch_st_idx[gen_batch_idx]
            curr_x_true_1_id = batch_curr_x_true_1[gen_batch_idx]
            curr_x_true_2_id = batch_curr_x_true_2[gen_batch_idx]
            curr_x_generated_1_id = batch_curr_x_generated_1[gen_batch_idx]
            curr_x_generated_2_id = batch_curr_x_generated_2[gen_batch_idx]
            total_sampled_tags = batch_total_sampled_tags[gen_batch_idx]
            total_sampled_cnt = batch_total_sampled_cnt[gen_batch_idx]

            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * ((update_count) / total_anneal_steps))
            else:
                anneal = anneal_cap

            update_count += 1
            end_idx = min(st_idx + BATCH_SIZE, N)
            # zzmask = get_mask(idxlist, 0, n_users, n_users)
            # print(end_idx,st_idx)
            # print(X.shape,mask.shape,curr_x_true_1_id.shape,curr_x_true_2_id.shape,curr_x_generated_1_id.shape,curr_x_generated_2_id.shape)
            feed_dict = {generator_network.input_ph: X,
                         # generator_network.user_mask_ph: mask,
                         all_user_emb: curr_all_user_emb,
                         # batch_user_ids: idxlist[st_idx:end_idx],
                         x_true_1_id: curr_x_true_1_id,
                         x_generated_1_id: curr_x_generated_1_id, x_true_2_id: curr_x_true_2_id,
                         x_generated_2_id: curr_x_generated_2_id, generated_tags: total_sampled_tags,
                         sampled_cnt: total_sampled_cnt, generator_network.keep_prob_ph: 0.75,
                         generator_network.is_training_ph: 1, generator_network.anneal_ph: anneal,
                         gen_lambda: curr_gen_lamda}
            feed_dict.update({placeholders['support'][i]: adj_matrix[i] for i in range(len(adj_matrix))})

        
            _, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2, batch_user_emb, curr_pair_weights,\
                curr_gen_out, curr_y_generated = sess.run(
                [g_trainer, g_loss_mean, g_vae_loss, gan_loss, user_emb, pair_weights, sampled_generator_out_non_zero,
                 y_generated],
                feed_dict=feed_dict)

            curr_all_user_emb[idxlist[st_idx:end_idx]] = batch_user_emb

        print("global-epoch:%s, generator-epoch:%s, g_loss:%.5f (vae_loss: %.5f + gan_loss: %.5f, anneal: %.5f)" % (
            i, j_gen, curr_g_loss, curr_g_loss_term_1, curr_g_loss_term_2, anneal))

    

    ########### MEMORY MODULE SUB EPOCHS ###########


    for j_mem in range(NUM_MEMORY_SUB_EPOCHS):
        for gen_batch_idx in indices:
            X = batch_X[gen_batch_idx]
            st_idx = batch_st_idx[gen_batch_idx]
            curr_x_true_1_id = batch_curr_x_true_1[gen_batch_idx]
            curr_x_true_2_id = batch_curr_x_true_2[gen_batch_idx]
            curr_x_generated_1_id = batch_curr_x_generated_1[gen_batch_idx]
            curr_x_generated_2_id = batch_curr_x_generated_2[gen_batch_idx]
            total_sampled_tags = batch_total_sampled_tags[gen_batch_idx]
            total_sampled_cnt = batch_total_sampled_cnt[gen_batch_idx]

            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * ((update_count) / total_anneal_steps))
            else:
                anneal = anneal_cap

            update_count += 1
            end_idx = min(st_idx + BATCH_SIZE, N)
            # zzmask = get_mask(idxlist, 0, n_users, n_users)
            # print(end_idx,st_idx)
            # print(X.shape,mask.shape,curr_x_true_1_id.shape,curr_x_true_2_id.shape,curr_x_generated_1_id.shape,curr_x_generated_2_id.shape)
            feed_dict = {generator_network.input_ph: X,
                         # generator_network.user_mask_ph: mask,
                         all_user_emb: curr_all_user_emb,
                         # batch_user_ids: idxlist[st_idx:end_idx],
                         x_true_1_id: curr_x_true_1_id,
                         x_generated_1_id: curr_x_generated_1_id, x_true_2_id: curr_x_true_2_id,
                         x_generated_2_id: curr_x_generated_2_id, generated_tags: total_sampled_tags,
                         sampled_cnt: total_sampled_cnt, generator_network.keep_prob_ph: 0.75,
                         generator_network.is_training_ph: 1, generator_network.anneal_ph: anneal,
                         gen_lambda: curr_gen_lamda}
            feed_dict.update({placeholders['support'][i]: adj_matrix[i] for i in range(len(adj_matrix))})
            _, curr_gan_loss = sess.run([memory_trainer, gan_loss], feed_dict=feed_dict)

        print("global-epoch:%s, memory-epoch:%s, gan_loss: %.5f)" % (i, j_mem, curr_gan_loss))



    ########### DISCRIMINATOR SUB EPOCHS ###########


    for j_disc in range(NUM_DISC_SUB_EPOCHS):
        for disc_batch_idx in indices:
            X = batch_X[disc_batch_idx]
            curr_x_true_1_id = batch_curr_x_true_1[disc_batch_idx]
            curr_x_true_2_id = batch_curr_x_true_2[disc_batch_idx]
            curr_x_generated_1_id = batch_curr_x_generated_1[disc_batch_idx]
            curr_x_generated_2_id = batch_curr_x_generated_2[disc_batch_idx]
            total_sampled_tags = batch_total_sampled_tags[disc_batch_idx]
            total_sampled_cnt = batch_total_sampled_cnt[disc_batch_idx]

            feed_dict = {generator_network.input_ph: X, x_true_1_id: curr_x_true_1_id,
                         x_generated_1_id: curr_x_generated_1_id, x_true_2_id: curr_x_true_2_id,
                         x_generated_2_id: curr_x_generated_2_id, generated_tags: total_sampled_tags,
                         sampled_cnt: total_sampled_cnt}

            feed_dict.update({placeholders['support'][i]: adj_matrix[i] for i in range(len(adj_matrix))})

            # print(feed_dict)

            _, curr_d_loss, y_tr, y_gen, x_g_1, x_g_2, e_matrix = sess.run(
                [d_trainer, d_loss, y_data, y_generated, x_gen_1, x_gen_2, emb_matrix],
                feed_dict=feed_dict)


        print("global-epoch:%s, discr-epoch:%s, d_loss:%.5f" % (i, j_disc, curr_d_loss))

    print('')




    ########### EVALUATE PERFORMANCE AT THE END OF EACH GLOBAL EPOCH ########### 



    X_vad = vad_data_tr[idxlist_vad[0:N_vad]]

    if sparse.isspmatrix(X_vad):
        X_vad = X_vad.toarray()
    X_vad = X_vad.astype('float32')
    mask = get_mask(idxlist_vad, 0, N_vad, n_users, offset=N)
    # pred_vad = sess.run(item_distribution_out,
    #                     feed_dict={generator_network.input_ph: X_vad, generator_network.user_mask_ph: mask,
    #                                batch_user_ids:idxlist_vad})
    pred_vad = sess.run(item_distribution_out,
                        feed_dict={generator_network.input_ph: X_vad, generator_network.user_mask_ph: mask})
    # exclude examples from training and validation (if any)
    pred_vad[X_vad.nonzero()] = -np.inf
    unique_item_count = overlap_at_k_batch(pred_vad, k=50)
    ndcg_vad_100 = ndcg_binary_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]])

    ndcg_vad_50 = ndcg_binary_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=50)

    recall_at_10, not_found_10 = recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=10)

    recall_at_20, not_found_20 = recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=20)

    recall_at_50, not_found_50 = recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=50)

    recall_at_100, not_found_100 = recall_at_k_batch(pred_vad, vad_data_te[idxlist_vad[0:N_vad]], k=100)

    # if np.mean(recall_at_50) > best_recall:
    #     best_recall = np.mean(recall_at_50)
    #     best_keys, best_M, best_prob = sess.run([keys, M, prob])

    best_recall_50 = max(best_recall_50, np.mean(recall_at_50))
    best_recall_20 = max(best_recall_20, np.mean(recall_at_20))
    best_recall_10 = max(best_recall_10, np.mean(recall_at_10))
    print('global-epoch:', i, 'gen-epoch:', j_gen, 'NDCG@50:', np.mean(ndcg_vad_50), 'NDCG@100: ',
          np.mean(ndcg_vad_100),
          'Recall@10:', np.mean(recall_at_10),
          'Recall@20:', np.mean(recall_at_20),
          'Recall@50:', np.mean(recall_at_50), 'Recall@100:', np.mean(recall_at_100), 'Num_users:', len(ndcg_vad_50),
          len(recall_at_20), len(recall_at_50),
          'best recall@50: ', best_recall_50, 'best recall@20: ', best_recall_20, 'best recall@10: ', best_recall_10, 'unique items: ', unique_item_count)

    print('')

    #####################       ADD SAVER CODE HERE TO SAVE GLOBAL CHECKPOINTS      ###########################################################################

