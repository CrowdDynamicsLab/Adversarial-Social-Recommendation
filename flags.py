import tensorflow as tf

# flags

# gpu_number = 3
base_path = "/home/ec2-user/Google_Local_Dataset/"


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
# del_all_flags(tf.flags.FLAGS)

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

# General params.
# Please refer to the train file for a description of the paramters 


flags.DEFINE_string('data_dir', "sample_dataset", 'dataset path')
flags.DEFINE_integer('gcn_feature_count', 128, 'GCN social features')
flags.DEFINE_float('batch_size_percentage', 0.05, 'Percentage of users per minibatch')
flags.DEFINE_float('user_sample_percentage', 0.05, 'Percentage of all possible pairs as fake pairs')

# For larger datasets use fewer global epochs and more sub epochs (since fake pair sampling might be quite expensive)
# Note that fake pair sampling happens at the start of each global epoch
flags.DEFINE_integer('global_epochs', 100, 'Number of global epochs')
flags.DEFINE_integer('gen_sub_epochs', 10, 'Number of generator sub epochs per global')
flags.DEFINE_integer('disc_sub_epochs', 5, 'Number of discriminator sub epochs per global')
flags.DEFINE_integer('mem_sub_epochs', 4, 'Number of memory sub epochs per global')

flags.DEFINE_float('d_learning_rate', 0.0032, 'Discriminator learning rate')
flags.DEFINE_float('g_learning_rate', 0.0005, 'Generator learning rate')
flags.DEFINE_float('m_learning_rate', 0.006, 'Memory (pair weight module) learning rate')

flags.DEFINE_float('ganlambda', 1.0, 'Adversary weight')
flags.DEFINE_float('mu', 2.0, 'Balance paramter for true and fake pairs in discriminator training')
flags.DEFINE_bool('use_dropout', True, 'Dropout')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate - set to 0 unless required')

flags.DEFINE_integer('memory_blocks', 10, 'Number of memory blocks for pair weighting')
flags.DEFINE_integer('d_hidden_1', 128, 'Hidden layer 1 of discriminator')
flags.DEFINE_integer('d_hidden_2', 128, 'Hidden layer 2 of discriminator')