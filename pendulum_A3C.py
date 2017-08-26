from Algorithm.A3C.A3C_continuous import *
import gym
from gym import wrappers
import argparse
import sys
import os
worker_threads = []
import threading
import multiprocessing


def main():
    
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    #parser.add_argument('--enable_noise_dense', dest='noisy', action='store_true', default=False)
    parser.add_argument('--num_workers', dest='num_workers',action='store',default=1,type=int)
    parser.add_argument('--testing', dest='test',action='store_true',default=False)
    args = parser.parse_args()

    env_to_use = 'Pendulum-v0'
    max_episode_length = 1000
    gamma = .995 # discount rate for advantage estimation and reward discounting
    
    # game parameters
    env = gym.make(env_to_use)
    s_size = int(np.prod(np.array(env.observation_space.shape))) 	# Get total number of dimensions in state
    a_size = int(np.prod(np.array(env.action_space.shape)))		# Assuming continuous action space
    env.seed(0)
    outdir = './Experiment/' +env_to_use+ '/A3C-agent-results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #env = wrappers.Monitor(env, outdir , force=True)
    np.random.seed(0)
    
    load_model = args.load_model
    #noisy = args.noisy
    num_workers = args.num_workers
    testing = args.test
    print(" num_workers = %d" % num_workers)
    #print(" noisy_net_enabled = %s" % str(noisy))
    print(" load_model = %s" % str(load_model))

    tf.reset_default_graph()
    
    model_path = './Experiment/'+env_to_use+'/A3C-agent-results/models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer_a = tf.train.AdamOptimizer(learning_rate=1e-4)
        trainer_c = tf.train.AdamOptimizer(learning_rate=1e-3)
        master_network = AC_Network(s_size,a_size,8,8,8,'global',None,None) # Generate global network
        num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        workers = []
            # Create worker classes
        for i in range(args.num_workers):
            worker = Worker(i,s_size,a_size,8,8,8,trainer_a,trainer_c,model_path,global_episodes,is_training= not testing, outdir = outdir)
            workers.append(worker)

        saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('loading Model succeeded')
        else:
            sess.run(tf.global_variables_initializer())
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            worker.start(env_to_use,outdir)
            t = threading.Thread(target=(worker_work))
            t.daemon = True
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-c received! Sending kill to threads...")
        for t in worker_threads:
            t.kill_received = True
