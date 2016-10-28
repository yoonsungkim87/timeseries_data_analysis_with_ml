def make_artificial_datasets(t_dur = 100, delay = [3,5], prt= False):
    import random
    import numpy as np
    
    buffers = [None for _ in range(len(delay))]
    for i in range(len(delay)):
        buffers[i] = [0] * delay[i]
        
    result = []
    for i in range(t_dur):
        temp_in = [None for _ in range(len(delay))]
        temp_out = 0
        for j in range(len(delay)):
            temp_in[j] = random.randrange(1,10)
            buffers[j].append(temp_in[j])
            temp_out += buffers[j].pop(0)
        result.append(temp_in + [temp_out])
        if prt:
            print('%2d'%i+' >>  '),
            for j in range(len(delay)):
                print('X%d'%j+': %2d'%temp_in[j]),
            print('Y: %2d'%temp_out)
    return np.array(result, dtype=np.int32)

def main():
    pass

if __name__ == '__main__':
    main()