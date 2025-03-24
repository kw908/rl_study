import os
import numpy as np
import tensorflow as tf
import keras
import gymnasium as gym
import imageio
import yaml

# === Env config ===
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only for stability

# === Load hyperparams ===
with open("ppo_config.yaml", "r") as f:
    config = yaml.safe_load(f)

steps_per_epoch = config["steps_per_epoch"]
hidden_sizes = tuple(config["hidden_sizes"])
ckptdir = config["ckptdir"]

# === Env with rendering ===
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
frames = []

# === Define exact same MLP ===
def mlp(x, sizes, activation=keras.activations.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = keras.layers.Dense(units=size, activation=activation)(x)
    return keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)

# === Create actor model exactly as in training ===
obs_input = keras.Input(shape=(obs_dim,), dtype="float32")
logits = mlp(obs_input, list(hidden_sizes) + [n_actions])
actor = keras.Model(inputs=obs_input, outputs=logits)

# === Force model to build before restore ===
_ = actor(tf.constant(np.zeros((1, obs_dim), dtype=np.float32)))

# === Restore checkpoint ===
ckpt = tf.train.Checkpoint(actor=actor)
latest_ckpt = tf.train.latest_checkpoint(ckptdir)
assert latest_ckpt, f"No checkpoint found in {ckptdir}"
ckpt.restore(latest_ckpt).expect_partial()
print(f"Restored checkpoint: {latest_ckpt}")

# === Evaluation policy (greedy deterministic) ===
@tf.function
def get_action(observation):
    logits = actor(observation)
    return tf.argmax(logits, axis=1)

# === Run one evaluation episode ===
obs, _ = env.reset()
done = False
episode_return = 0
step = 0
max_steps = 1000

while not done and step < max_steps:
    frame = env.render()
    frames.append(frame)

    obs_tensor = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)
    action = get_action(obs_tensor).numpy()[0]
    obs, reward, done, _, _ = env.step(action)
    episode_return += reward
    step+=1

env.close()
print(f"Episode finished. Total return: {episode_return}")

# === Save as GIF ===
output_path = "ppo_eval_episode_2.gif"
imageio.mimsave(output_path, frames, fps=30)
print(f"Saved episode to: {output_path}")
