import gym
import neat
import os
import visualize
import numpy as np

# Алгоритм сам определяет сколько входов и сколько выходов требуется
# пока работает только с дискретными входами спортзала

"""
gen - Номер енерации
env - спортзал 
envs - []]

 - геномы
ge = []
net - сеть 
nets = []
cart - ответ сети
carts = []
 - Маска, показывает какие геномы еще живы
do_it = [] 


"""

GYM_NAME = 'CartPole-v1' # CartPole-v1
                            # LunarLander-v2
                            # LunarLanderContinuous-v2
                            # MountainCar-v0
                            # Acrobot-v1
gen = 0
envs = [] #создаю гломбально т.к. если постоянно пересоздавать спортзал, он ломается

def test_winer(env,winner,config,protc=50,testov=3,render=False, debug=False, print_genome=False):

    # env = gym.make(GYM_NAME)
    ge = winner
    # config.pop_size = 100
    net = neat.nn.FeedForwardNetwork.create(ge, config)
    # env.reset()


    def del_files(files):
        for f in files:
            if os.path.exists(f):
                print(f)
                os.remove(f)

    if print_genome:
        del_files(['Digraph.gv','Digraph.gv.pdf','Digraph.gv.png','Digraph.gv.svg'])
        visualize.draw_net(config, ge, view=True)


    rez=0
    for i in range(testov):
        score=0
        observation = env.reset()
        action=0
        done = False

        pravka=1
        # observation, reward, done, info = env.step(action)
        while not done:
            observation = np.concatenate((observation,[(pravka-observation[0])/3]))
            output = net.activate(observation)
            #action=ge.front(*observation)
            if env.action_space.dtype.name == 'int64':
                action = output.index(max(output)) # Выбираю вход с самым сильным сигналом

            observation, reward, done, info = env.step(action)
            # print(observation)
            # score+=reward


            # Ввожу поправку на точку, к которой должен стремитсья агент
            # if pravka==0:  # Выбираю какой тип наград и штрафов использовать
            #     score += reward # Награждаем
            #     score -= abs(pravka-observation[0])/3 # Штраф за отклонение от центра
            # else:
            if True:
                score += reward/10
                if score>100: # Если прошел 3 чекпоинта, награждаем за жизнь
                    score += reward # Награждаем
                # Если агент приближается к заданной точке на погрешность 0.01, выставляем ему новую точку
                if abs(pravka-observation[0])<0.01:
                    score += reward+50 # Награждаем только при прохождении чекпоинтов
                    if observation[0]>0:
                        pravka=-abs(pravka)
                    else:
                        pravka=abs(pravka)




            #
            #
            # if abs(pravka-observation[0])<0.01:
            #     if observation[0]>0:
            #         pravka=-abs(pravka)
            #         ge.fitness += reward+50 # Награждаем только при прохождении чекпоинтов
            #     else:
            #         pravka=abs(pravka)
            #         ge.fitness += reward+50 # Награждаем только при прохождении чекпоинтов


            if render:
                env.render()
        if debug:
            print(f'Игра №{i+1} из {testov} Счет = {score}')
        rez+=score
    # env.close()
    if debug:
        print(f'Средний чет = {round(100/config.fitness_threshold*rez/testov,0)}%  Порог прохождения {protc}%')

    return 100/config.fitness_threshold*rez/testov>=protc

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    global envs

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Выясняю у спортзала число входов и выходов, и правлю конфиг до создания популяции
    envs.append(gym.make(GYM_NAME))
    config.genome_config.num_outputs = envs[0].action_space.n # Число выходов сети рано числу входов спортзала
    config.genome_config.output_keys=[i for i in range(config.genome_config.num_outputs)] # Нумерою выходы
    config.genome_config.num_inputs = envs[0].observation_space.shape[0]+1 # Число входов сети равно числу выходов спортзала # лишний вход, это расстояние от тележки до чекпоинта деленное на 3
    config.genome_config.input_keys = [-i-1 for i in range(config.genome_config.num_inputs)]
    config.fitness_threshold = envs[0].spec.reward_threshold # максимальное состижение в данном спортзале

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-2099')
    #
    #
    # # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # neat.checkpoint

    f_exit = False
    while not f_exit:
        winner = p.run(eval_genomes, 2000)  # Максимальное число генераций

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))



        # Если 3 игры средний результат за 3 игры привысит 90% от максимума, выходим
        f_exit=test_winer(env=envs[0],winner=winner,config=config, protc=90, testov=3,render=False, debug=True, print_genome=True)





    # Сохраняю популяции
    a=neat.Checkpointer()
    a.save_checkpoint(config=config,population=p,species_set=p.species,generation=p.generation)
    # neat.Checkpointer.save_checkpoint(config=config,population=p,species_set=p.species,generation=p.generation)
    # a=neat.checkpoint.Population(config=config)

    # тут я делаю следующее, когда найден подходящий мне алгоритм, хочу стобы он играл бесконечно
    while True:
        test_winer(env=envs[0],winner=winner,config=config, protc=90, testov=100, render=True, debug=True)
    for i in range(len(envs)):
        envs[0].close()
        envs.pop(0)



def eval_genomes(genomes, config):
    global gen
    global envs

    gen += 1

    # envs = []  # среды с агентами
    ge = []  # геномы
    nets = []
    carts = []
    # graph_net = []
    do_it = [] # Маска, показывает какие геномы еще живы

    # def viz(genome):
    #     visualize.draw_net(config=config,genome=genome, view=False)
    #     graph_net = pygame.image.load('Digraph.gv.png')
    #     graph_net.set_colorkey((255, 255, 255))
    #     graph_net = graph_net.convert_alpha()
    #
    #     # Во избежание ошибок, проверяю если файлы существуют, то удаляю их
    #     def del_files(files):
    #         for f in files:
    #             if os.path.exists(f):
    #                 print(f)
    #                 os.remove(f)
    #
    #     del_files(['Digraph.gv','Digraph.gv.pdf','Digraph.gv.png','Digraph.gv.svg'])
    #     return graph_net



    # создаем среды
    for genome_id, genome in genomes:
        do_it.append(True)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        # graph_net.append(viz(genome))

    # Если спортзалы еще не созданы, создаем их
    # if len(envs)==0:
    #     for i in range(len(genomes)):
    #         envs.append(gym.make('CartPole-v1'))  # CartPole-v1  LunarLander-v2 LunarLanderContinuous-v2
    for i in range(len(ge)):
        if len(envs)-1-i<0:
            envs.append(gym.make(GYM_NAME))  # CartPole-v1  LunarLander-v2 LunarLanderContinuous-v2

    if gen == 1:
        print("action space: {0!r}".format(envs[0].action_space))
        print("observation space: {0!r}".format(envs[0].observation_space))


    # for env in envs:
    #     env.reset()


    score = 0
    f_run = True

    for i in range(len(ge)):
        action = 0#[0.0,0.0]
        # observation, reward, done, info = envs[i].step(action)  # take a random action
        observation = envs[i].reset()
        # CarRacing-v0
        # action = array[0.0,0.0,0.0]
        # s, r, done, info
        carts.append({'observation': observation})

    # print(f'len(envs)={len(envs)}')
    # print(f'len(ge)={len(ge)}')
    # print(f'len(nets)={len(nets)}')
    # print(f'len(carts)={len(carts)}')
    # print(f'len(do_it)={len(do_it)}')


    # Отклонение от центра, к этой точке будет стремиться агент. Или штраф :-)
    pravka=[1 for i in range(len(ge))]

    while f_run and len(ge) > 0:
        f_run = False
        score += 1

        # Запрашиваем дествие от сети
        for n in range(len(ge)):
            if do_it[n]:
                # if ge[n].fitness>500:
                #     envs[n].render()
                # envs[n].render()
                # запрашиваем дейсвие от сети
                temp=(carts[n]['observation'])
                temp = np.concatenate((temp,[(pravka[n]-carts[n]['observation'][0])/3])) #Записываю расстояние до чекпоинта на 4-ый вход сети
                output = nets[n].activate(temp)
                if envs[n].action_space.dtype.name == 'int64':
                    action = output.index(max(output)) # Выбираю вход с самым сильным сигналом

                observation, reward, done, info = envs[n].step(action)  # take a random action
                carts[n] = {'observation': observation, 'reward': reward, 'done': done, 'info': info}


                # # Ввожу поправку на точку, к которой должен стремитсья агент
                # if pravka[n]==0:  # Выбираю какой тип наград и штрафов использовать
                #     ge[n].fitness += reward # Награждаем
                #     ge[n].fitness -= abs(pravka[n]-carts[n]['observation'][0])/3 # Штраф за отклонение от центра
                # else:
                if True:
                    ge[n].fitness += reward/10
                    if ge[n].fitness>100: # Если прошел 3 чекпоинта, награждаем за жизнь
                        ge[n].fitness += reward # Награждаем
                    # Если агент приближается к заданной точке на погрешность 0.01, выставляем ему новую точку
                    if abs(pravka[n]-carts[n]['observation'][0])<0.01:
                        ge[n].fitness += reward+50 # Награждаем только при прохождении чекпоинтов
                        if carts[n]['observation'][0]>0:
                            pravka[n]=-abs(pravka[n])
                        else:
                            pravka[n]=abs(pravka[n])

        for i in range(len(ge)):
            if carts[i]['done']:
                do_it[i] = False

        best_ge_fitness=0 # Номер Генома с лучшим результатом в этом поколении
        for i in range(len(ge)):
            if ge[i].fitness>ge[best_ge_fitness].fitness:
                best_ge_fitness = i

        for i in range(len(do_it)):
            if do_it[i]:
                f_run = True

    # Смотрим на игру лучшего генома в этом поколении (каждое 10 поколение)
    if gen % 10 == 1:
        test_winer(env=envs[0],winner=ge[best_ge_fitness],config=config,protc=50,testov=1, render=True,print_genome=True)

    for i in range(len(do_it)):
        ge.pop(0)
        nets.pop(0)
        carts.pop(0)

# for env in envs:
    #     env.close()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    try:
        run(config_path)
    except Exception as e:
        print('!!!!!!!!!!!!!!!!!',str(e))

