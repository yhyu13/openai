import numpy as np
import scipy.signal as ss
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from gym import wrappers

from time import sleep
from time import time
from time import gmtime, strftime

# ================================================================
# Helper function
# ================================================================

# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Normalize state 
def normalize(s):
    s = np.asarray(s)
    s = (s-np.mean(s)) / np.std(s)
    return s


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def discount(x, gamma):
    return ss.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# ================================================================
# Model components
# ================================================================

# Actor Network------------------------------------------------------------------------------------------------------------
class AC_Network():
    def __init__(self,s_size,a_size,layer1,layer2,layer3,scope,trainer_a,trainer_c):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            
	        # Create the model, use the default arg scope to configure the batch norm parameters.
            '''
            conv1 = tf.nn.elu(tf.nn.conv1d(self.imageIn,tf.truncated_normal([2,1,8],stddev=0.1),2,padding='VALID'))
            conv2 = tf.nn.elu(tf.nn.conv1d(conv1,tf.truncated_normal([3,8,16],stddev=0.05),1,padding='VALID'))
        
            hidden = slim.fully_connected(slim.flatten(conv2),200,activation_fn=tf.nn.elu)'''
            
            hidden1 = slim.fully_connected(self.inputs,layer1,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = slim.fully_connected(hidden1,layer2,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
            hidden3 = slim.fully_connected(hidden2,layer3,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
            
            hidden1_c = slim.fully_connected(self.inputs,layer1,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
            hidden2_c = slim.fully_connected(hidden1_c,layer2,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
            hidden3_c = slim.fully_connected(hidden2_c,layer3,activation_fn=tf.nn.elu,weights_initializer=tf.contrib.layers.xavier_initializer())
    
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden3,a_size,activation_fn=tf.nn.softmax,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=None)

            self.value = slim.fully_connected(hidden3_c,1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=None)
                
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                #Loss functions
                self.responsible_action = tf.reduce_sum(self.policy * self.actions_onehot,[1])
                
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy =  -tf.reduce_sum(tf.log(self.policy)*self.policy)  # encourage exploration

                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_action)*self.advantages) - 0.01 * self.entropy

                self.loss = self.value_loss + self.policy_loss 

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients_a = tf.gradients(self.policy_loss,local_vars)
                self.gradients_c = tf.gradients(self.value_loss,local_vars)
                
                #self.var_norms = tf.global_norm(local_vars)
                #self.gradients_a,self.grad_norms = tf.clip_by_global_norm(self.gradients_a,40.0)
                #self.gradients_c,self.grad_norms = tf.clip_by_global_norm(self.gradients_c,40.0)
                
                #Apply local gradients to global network
                #Comment these two lines out to stop training
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads_a = trainer_a.apply_gradients(zip(self.gradients_a,global_vars))
                self.apply_grads_c = trainer_c.apply_gradients(zip(self.gradients_c,global_vars))
                
# Learning to run Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,s_size,a_size,layer1,layer2,layer3,trainer_a,trainer_c,model_path,global_episodes,is_training,outdir):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer_a = trainer_a
        self.trainer_c = trainer_c
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(outdir+"/train_"+str(self.number))
        self.is_training = is_training
        self.outdir = outdir

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,layer1,layer2,layer3,self.name,trainer_a,trainer_c)
        self.update_local_ops = update_target_graph('global',self.name)
        
    def start(self,env_to_use,outdir):
            # game parameters # due to Async nature, each agent needs to start its own gym env.
            self.env = gym.make(env_to_use)
            self.env.seed(0)
            if self.name == 'worker_1':
                self.env = wrappers.Monitor(self.env, outdir , force=True)
            np.random.seed(0)

        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
    	# reward clipping:  scale and clip the values of the rewards to the range -1,+1
    	#rewards = (rewards - np.mean(rewards)) / np.max(abs(rewards))

        next_observations = rollout[:,3] # Aug 1st, notice next observation is never used
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        l,v_l,p_l,e_l,_,_ = sess.run([self.local_AC.loss,self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            #self.local_AC.grad_norms,
            self.local_AC.apply_grads_a,self.local_AC.apply_grads_c],
            feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout),p_l / len(rollout),e_l / len(rollout)
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        if self.is_training:
            episode_count = sess.run(self.global_episodes)
        else:
            episode_count = 0
        wining_episode_count = 0
        total_steps = 0
        print ("Starting worker " + str(self.number))
        outdir = self.outdir
        with open(outdir+'/result.txt','w') as f:
            f.write(strftime("Starting time: %a, %d %b %Y %H:%M:%S\n", gmtime()))

        with sess.as_default(), sess.graph.as_default():
            #not_start_training_yet = True
            while not coord.should_stop():
                        
                sess.run(self.update_local_ops)
		        #sess.run(self.update_local_ops_target)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                s = self.env.reset()
                if self.name == 'worker_1' and episode_count % 5 == 0:
                    self.env.render()
                #s = normalize(s)
                #st = time()
                while done == False:
                    #Take an action

                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
                                feed_dict={self.local_AC.inputs:[s]})
                                
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    ob,r,done,_ = self.env.step(a)
                    if self.name == 'worker_1' and episode_count % 5 == 0:
                        self.env.render()
                    '''
                    if self.name == 'worker_0':
                        ct = time()
                        print(ct-st)
                        st = ct
                    '''
                    if done == False:
                        s1 = ob
                    else:
                        s1 = s
                    #s1 = normalize(s)
                    '''
                    if self.name == 'worker_1':
                        #print(done)
                        #print(s1)
                        print(s)
                        print(a)
                        print(r)    '''
                    #r = np.maximum(r,0.0)
                    episode_buffer.append([s,a,r,s1,done,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                            
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                       
                    if len(episode_buffer) == 30 and done != True and episode_step_count != max_episode_length - 1: # change pisode length to 5, and try to modify Worker.train() function to utilize the next frame to train imagined frame.
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        if self.is_training:
                            v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                            l,v_l,p_l,e_l = self.train(episode_buffer,sess,gamma,v1)
                            sess.run(self.update_local_ops)
                            episode_buffer = []
                    
                    if done == True:
                        #print(episode_step_count)
                        break
                           
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                    
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) > 0:
                    if self.is_training:
                        l,v_l,p_l,e_l = self.train(episode_buffer,sess,gamma,0.0)
                        #print(l,v_l,p_l,e_l,g_n)
	                    #sess.run(self.update_global_target)
                                    
                        
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    if self.is_training:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=np.sum(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=np.sum(e_l))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                            
                    if self.name == 'worker_0':
                        print(("Episodes "+str(episode_count)+" mean reward (training): %.2f\n" % mean_reward))
                        with open(outdir+'/result.txt','a') as f:
                            f.write("Episodes "+str(episode_count)+" mean reward (training): %.2f\n" % mean_reward)

                        if episode_count % 100 == 0 and self.is_training:
                            saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                            print(("Saved Model at episode: "+str(episode_count)+"\n"))
                            with open(outdir+'/result.txt','a') as f:
                                f.write("Saved Model at episode: "+str(episode_count)+"\n")
                    
                if self.name == 'worker_0' and self.is_training:
                    sess.run(self.increment)
                
                if self.name == 'worker_0':
                    print('episode {}'.format(episode_count))        
                episode_count += 1
                
                '''    
                if self.name == "worker_1" and episode_reward > 2.:
                    wining_episode_count += 1
                    print('Worker_1 is stepping forward in Episode {}! Reward: {:.2f}. Total percentage of success is: {}%'.format(episode_count, episode_reward, int(wining_episode_count / episode_count * 100)))
                    with open('result.txt','a') as f:
                    f.wirte('Worker_1 is stepping forward in Episode {}! Reward: {:.2f}. Total percentage of success is: {}%\n'.format(episode_count, episode_reward, int(wining_episode_count / episode_count * 100)))'''
        
        # All done Stop trail
        # Confirm exit
        print('Exit/Done '+self.name)
