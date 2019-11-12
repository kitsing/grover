import tensorflow as tf


def restore(scope, checkpoint, sess):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    checkpoint_vars = tf.train.list_variables(checkpoint)
    checkpoint_names = [_[0] for _ in checkpoint_vars]
    assignment_map = dict()
    unused_vars_in_checkpoint = set(checkpoint_names)
    for var in vars:
        name = var.name
        assert name.endswith(':0')
        name = name[:-2]

        # hack
        if name.startswith('discriminator_final_layer'):
            if name in unused_vars_in_checkpoint:
                assignment_map[name] = var
            else:
                tf.logging.warn(f'key not found: {name}')
            continue

        splitted_name = name.split(scope)
        if len(splitted_name) > 1:
            # new_name = ''.join(['newslm'] + splitted_name[1:])
            new_name = 'newslm'.join(splitted_name)
            if new_name in unused_vars_in_checkpoint:
                assignment_map[new_name] = var
                tf.logging.info(f'key found: {new_name} -> {name}')
                unused_vars_in_checkpoint.remove(new_name)
            else:
                tf.logging.warn(f'key not found: {new_name}')
        else:
            tf.logging.warn(f'key {name} does not start with {scope}')

    tf.logging.warn(f'unused variables in checkpoint: {unused_vars_in_checkpoint}')
    # print(gen_assignment_map)
    saver = tf.train.Saver(var_list=assignment_map)
    saver.restore(sess, checkpoint)