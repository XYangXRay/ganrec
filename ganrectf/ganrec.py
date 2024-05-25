import os
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
from ganrectf.propagators import TomoRadon, TensorRadon, PhaseFresnel, PhaseFraunhofer
from ganrectf.models import make_generator, make_discriminator
from ganrectf.utils import RECONmonitor, ffactor


def tf_configures():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load the configuration from the JSON file
def load_config(filename):
    # Get the directory of the script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the full path to the config file
    config_path = os.path.join(dir_path, filename)

    with open(config_path, "r") as file:
        config = json.load(file)
    return config


# Use the configuration
config = load_config("config.json")


# @tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    )
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    )
    total_loss = real_loss + fake_loss
    return total_loss


def l1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(img1 - img2))


def l2_loss(img1, img2):
    return tf.square(tf.reduce_mean(tf.abs(img1 - img2)))


# @tf.function
def generator_loss(fake_output, img_output, pred, l1_ratio):
    gen_loss = (
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output))
        )
        + l1_loss(img_output, pred) * l1_ratio
    )
    return gen_loss


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img


def avg_results(recon, loss):
    sort_index = np.argsort(loss)
    recon_tmp = recon[sort_index[:10], :, :, :]
    return np.mean(recon_tmp, axis=0)


class GANtomo:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config["GANtomo"]
        tomo_args.update(**kwargs)
        super(GANtomo, self).__init__()
        tf_configures()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = tomo_args["iter_num"]
        self.conv_num = tomo_args["conv_num"]
        self.conv_size = tomo_args["conv_size"]
        self.dropout = tomo_args["dropout"]
        self.l1_ratio = tomo_args["l1_ratio"]
        self.g_learning_rate = tomo_args["g_learning_rate"]
        self.d_learning_rate = tomo_args["d_learning_rate"]
        self.save_wpath = tomo_args["save_wpath"]
        self.init_wpath = tomo_args["init_wpath"]
        self.init_model = tomo_args["init_model"]
        self.recon_monitor = tomo_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.prj_input.shape[0], self.prj_input.shape[1], self.conv_num, self.conv_size, self.dropout, 1
        )

        self.discriminator = make_discriminator(self.prj_input.shape[0], self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    @tf.function
    def tfnor_tomo(self, img):
        img = tf.image.per_image_standardization(img)
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    @tf.function
    def recon_step(self, prj, ang):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            tomo_radon_obj = TomoRadon(recon, ang)
            prj_rec = tomo_radon_obj.compute()
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = self.tfnor_tomo(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath + "generator.h5")
            print("generator is initilized")
            self.discriminator.load_weights(self.init_wpath + "discriminator.h5")
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo")
            recon_monitor.initial_plot(self.prj_input)
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            step_result = self.recon_step(prj, ang)
            recon[epoch, :, :, :] = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]
            ###########################################################################
            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=step_result["d_loss"].numpy())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    prj_rec = np.reshape(step_result["prj_rec"], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        if self.recon_monitor:
            recon_monitor.close_plot()
        return recon[epoch].astype(np.float32)


class GANtensor:
    def __init__(self, prj_input, angle, psi, **kwargs):
        tomo_args = config["GANtensor"]
        tomo_args.update(**kwargs)
        super(GANtensor, self).__init__()
        tf_configures()
        self.prj_input = prj_input
        self.angle = angle
        self.psi = psi
        self.iter_num = tomo_args["iter_num"]
        self.conv_num = tomo_args["conv_num"]
        self.conv_size = tomo_args["conv_size"]
        self.dropout = tomo_args["dropout"]
        self.l1_ratio = tomo_args["l1_ratio"]
        self.g_learning_rate = tomo_args["g_learning_rate"]
        self.d_learning_rate = tomo_args["d_learning_rate"]
        self.save_wpath = tomo_args["save_wpath"]
        self.init_wpath = tomo_args["init_wpath"]
        self.init_model = tomo_args["init_model"]
        self.recon_monitor = tomo_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.prj_input.shape[0], self.prj_input.shape[1], self.conv_num, self.conv_size, self.dropout, 6
        )
        self.discriminator = make_discriminator(self.prj_input.shape[0], self.prj_input.shape[1])
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    @tf.function
    def tfnor_tomo(self, data):
        # Calculate the mean and standard deviation of the data
        mean = tf.reduce_mean(data)
        std = tf.math.reduce_std(data)
        # Standardize the data (z-score normalization)
        standardized_data = (data - mean) / std
        # Find the minimum value in the standardized data
        standardized_min = tf.reduce_min(standardized_data)
        # Shift the data to start from 0
        shifted_data = standardized_data - standardized_min
        return shifted_data

    # def tfnor_tomo(self, img):
    #     img = tf.image.per_image_standardization(img)
    #     img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    #     return img

    @tf.function
    def recon_step(self, prj, ang, psi):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            tomo_radon_obj = TensorRadon(recon, ang, psi)
            prj_rec = tomo_radon_obj.compute()
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = self.tfnor_tomo(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        psi = self.psi
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath + "generator.h5")
            print("generator is initilized")
            self.discriminator.load_weights(self.init_wpath + "discriminator.h5")
        recon = np.zeros((self.iter_num, px, px, 6))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo")
            recon_monitor.initial_plot(self.prj_input)
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):
            step_result = self.recon_step(prj, ang, psi)
            recon[epoch, :, :, :] = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]

            ###########################################################################
            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=step_result["d_loss"].numpy())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if recon_monitor:
                    prj_rec = np.reshape(step_result["prj_rec"], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch, :, :, 0], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        recon_monitor.close_plot()
        recon_out = np.transpose(recon[epoch], axes=(2, 0, 1))
        return recon_out.astype(np.float32)


class GANtomo3D:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config["GANphase"]
        tomo_args.update(**kwargs)
        super(GANtomo3D, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = tomo_args["iter_num"]
        self.conv_num = tomo_args["conv_num"]
        self.conv_size = tomo_args["conv_size"]
        self.dropout = tomo_args["dropout"]
        self.l1_ratio = tomo_args["l1_ratio"]
        self.g_learning_rate = tomo_args["g_learning_rate"]
        self.d_learning_rate = tomo_args["d_learning_rate"]
        self.save_wpath = tomo_args["save_wpath"]
        self.init_wpath = tomo_args["init_wpath"]
        self.init_model = tomo_args["init_model"]
        self.recon_monitor = tomo_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.prj_input.shape[0], self.prj_input.shape[1], self.conv_num, self.conv_size, self.dropout, 1
        )
        self.discriminator = make_discriminator(self.prj_input.shape[0], self.prj_input.shape[1])
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    def tfnor_tomo(data):
        # Calculate the mean and standard deviation of the data
        mean = tf.reduce_mean(data)
        std = tf.math.reduce_std(data)

        # Standardize the data (z-score normalization)
        standardized_data = (data - mean) / std

        # Find the minimum value in the standardized data
        standardized_min = tf.reduce_min(standardized_data)

        # Shift the data to start from 0
        shifted_data = standardized_data - standardized_min

        return shifted_data

    @tf.function
    def recon_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            prj_rec = self.tomo_radon(recon, ang)
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = self.tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = self.tfnor_data(recon)
            prj_rec = TomoRadon(recon, ang).compute
            prj_rec = self.tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_filter": prj_filter, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        # prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath + "generator.h5")
            print("generator is initilized")
            self.discriminator.load_weights(self.init_wpath + "discriminator.h5")
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo")
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[: epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result["prj_rec"], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print(
                    "Iteration {}: G_loss is {} and D_loss is {}".format(
                        epoch + 1, gen_loss[epoch], step_result["d_loss"].numpy()
                    )
                )
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        return recon[epoch]
        # return avg_results(recon, gen_loss)


class GANphase:
    def __init__(self, i_input, energy, z, pv, **kwargs):
        phase_args = config["GANphase"]
        phase_args.update(**kwargs)
        super(GANphase, self).__init__()
        tf_configures()
        self.i_input = i_input
        self.px, _ = i_input.shape
        self.energy = energy
        self.z = z
        self.pv = pv
        self.iter_num = phase_args["iter_num"]
        self.conv_num = phase_args["conv_num"]
        self.conv_size = phase_args["conv_size"]
        self.dropout = phase_args["dropout"]
        self.l1_ratio = phase_args["l1_ratio"]
        self.abs_ratio = phase_args["abs_ratio"]
        self.g_learning_rate = phase_args["g_learning_rate"]
        self.d_learning_rate = phase_args["d_learning_rate"]
        self.phase_only = phase_args["phase_only"]
        self.save_wpath = phase_args["save_wpath"]
        self.init_wpath = phase_args["init_wpath"]
        self.init_model = phase_args["init_model"]
        self.recon_monitor = phase_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.i_input.shape[0], self.i_input.shape[1], self.conv_num, self.conv_size, self.dropout, 2
        )
        self.discriminator = make_discriminator(self.i_input.shape[0], self.i_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-3)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    @tf.function
    def rec_step(self, i_input, ff):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            phase = tfnor_phase(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.px, self.px])
            absorption = (1 - tfnor_phase(recon[:, :, :, 1])) * self.abs_ratio
            absorption = tf.reshape(absorption, [self.px, self.px])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            phase_obj = PhaseFresnel(phase, absorption, ff, self.px)
            i_rec = phase_obj.compute()
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"phase": phase, "absorption": absorption, "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        ff = ffactor(self.px * 2, self.energy, self.z, self.pv)
        i_input = np.reshape(self.i_input, (1, self.px, self.px, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        phase = np.zeros((self.iter_num, self.px, self.px))
        absorption = np.zeros((self.iter_num, self.px, self.px))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("phase")
            recon_monitor.initial_plot(self.i_input)
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(i_input, ff)
            phase[epoch, :, :] = step_results["phase"]
            absorption[epoch, :, :] = step_results["absorption"]
            i_rec = step_results["i_rec"]
            gen_loss[epoch] = step_results["g_loss"]
            d_loss = step_results["d_loss"]
            ###########################################################################
            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=d_loss.numpy())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if recon_monitor:
                    i_rec = np.reshape(i_rec, (self.px, self.px))
                    i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.px))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
        recon_monitor.close_plot()
        return absorption[epoch].astype(np.float32), phase[epoch].astype(np.float32)


class GANdiffraction:
    def __init__(self, i_input, mask, **kwargs):
        diffraction_args = config["GANdiffraction"]
        diffraction_args.update(**kwargs)
        super(GANdiffraction, self).__init__()
        self.i_input = i_input
        self.mask = mask
        self.px, _ = i_input.shape
        self.iter_num = diffraction_args["iter_num"]
        self.conv_num = diffraction_args["conv_num"]
        self.conv_size = diffraction_args["conv_size"]
        self.dropout = diffraction_args["dropout"]
        self.l1_ratio = diffraction_args["l1_ratio"]
        self.abs_ratio = diffraction_args["abs_ratio"]
        self.g_learning_rate = diffraction_args["g_learning_rate"]
        self.d_learning_rate = diffraction_args["d_learning_rate"]
        self.phase_only = diffraction_args["phase_only"]
        self.save_wpath = diffraction_args["save_wpath"]
        self.init_wpath = diffraction_args["init_wpath"]
        self.init_model = diffraction_args["init_model"]
        self.recon_monitor = diffraction_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    @tf.function
    def tfnor_diff(img):
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    def make_model(self):
        self.generator = make_generator(
            self.i_input.shape[0], self.i_input.shape[1], self.conv_num, self.conv_size, self.dropout, 2
        )
        self.discriminator = make_discriminator(self.i_input.shape[0], self.i_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-3)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    @tf.function
    def rec_step(self, i_input):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            phase = self.tfnor_diff(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.px // 2, self.px // 2])
            phase = tf.pad(phase, [[64, 64], [64, 64]])
            absorption = (1 - self.tfnor_diff(recon[:, :, :, 1])) * self.abs_ratio
            absorption = tf.reshape(absorption, [self.px // 2, self.px // 2])
            absorption = tf.pad(absorption, [[64, 64], [64, 64]])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            phase_obj = PhaseFraunhofer(phase, absorption)
            i_rec = phase_obj.compute()
            mask = tf.reshape(self.mask, [1, self.mask.shape[0], self.mask.shape[1], 1])
            if self.mask:
                i_rec = tf.multiply(i_rec, mask)
            i_rec = self.tfnor_diff(i_rec)
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"phase": phase, "absorption": absorption, "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        i_input = np.reshape(self.i_input, (1, self.px, self.px, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        phase = np.zeros((self.iter_num, self.px, self.px))
        absorption = np.zeros((self.iter_num, self.px, self.px))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("phase")
            recon_monitor.initial_plot(self.i_input)
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(i_input)
            phase[epoch, :, :] = step_results["phase"]
            absorption[epoch, :, :] = step_results["absorption"]
            i_rec = step_results["i_rec"]
            gen_loss[epoch] = step_results["g_loss"]
            d_loss = step_results["d_loss"]
            ###########################################################################

            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=d_loss.numpy())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    i_rec = np.reshape(i_rec, (self.px, self.px))
                    i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.px))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
        if self.recon_monitor:
            recon_monitor.close_plot()
        return absorption[epoch], phase[epoch]
