#@title option loader
from imports import *
def read_file(path):
		''' We just read the file and put its contents in fstr.'''
		fstr = ''
		file = open(path, 'r')
		for line in file:
			fstr += line

		return fstr

def load_option(env, fstr=None, filePath=None):
    ''' I now parse the received string. I'll store everything in a matrix
    (matrixMDP) such that U, D, R, L, T mean 'up', 'down', 'right', 'left',
    'terminate', referring to the proper action to be taken.'''
    if fstr is None:
        fstr = read_file(filePath)
    data = fstr.split('\n')
    numRows = int(data[0].split(',')[0])
    numCols = int(data[0].split(',')[1])
    matrixMDP = np.zeros((numRows, numCols), dtype = np.int)
    option_policy = np.zeros((env._num_states, env._num_actions))
    option_interest = np.ones((env._num_states,))
    for i in range(len(data) - 1):
        for j in range(len(data[i+1])):
            s = env._get_state_idx(i, j)

            if data[i+1][j] == 'X':
                matrixMDP[i][j] = 4 #terminate
                option_interest[s] = 0
            elif data[i+1][j] == 'L':
                matrixMDP[i][j] = 3 #left
                option_policy[s][3] = 1
            elif data[i+1][j] == 'D':
                matrixMDP[i][j] = 2 #down
                option_policy[s][2] = 0.5
                option_policy[s][3] = 0.5
            elif data[i+1][j] == 'R':
                matrixMDP[i][j] = 1 #right
                option_policy[s][1] = 1
            elif data[i+1][j] == 'U':
                matrixMDP[i][j] = 0 #up
                option_policy[s][0] = 0.5
                option_policy[s][1] = 0.5

    # option = []
    # for i in range(numRows):
    #     for j in range(numCols):
    #         option.append(matrixMDP[i][j])

    return option_interest, option_policy