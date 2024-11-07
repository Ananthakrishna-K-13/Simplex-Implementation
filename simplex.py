# IMT2022086 - Ananthakrishna K
# IMT2022103 - Anurag Ramaswamy
# IMT2022115 - Kandarp Dave

import numpy as np
inf = 1000000000000000000000000000000

redundant = [] # To store the redundant constraints

# Get input in list format
def getlineip(s):
    ret = []
    for ind in range(len(s)):
        i = s[ind]
        if(i=='' or i=='\n'):
            continue
        if '@' in i:
            lit = i.split('@')
            if(int(lit[1]) != ind+1):
                print("INPUT ERROR")
                exit()
            ret.append(lit[0])
        else:
            ret.append(i)

    ret = list(map(float,ret))
    return ret 

# Remove redundant equality constraints
def remove_redundant(equality_constraints, h, d, slack):

    if len(equality_constraints) == 0:
        return [], []

    eq_prime = equality_constraints[:, 0:d - slack]
    h_prime = np.reshape(h, newshape=(-1, 1))
    augmented = np.hstack((eq_prime, h_prime)) # Augmented matrix [eq_prime | h_prime]

    mat = []
    lin_independent = [] # To store the linearly independent rows of augmented matrix
    rank = 0 # Store rank of matrix mat

    for row in augmented:
        
        mat.append(row)
        newrank = np.linalg.matrix_rank(mat) # Find rank after appending new row

        # New row is linearly independent
        if newrank == rank + 1:
            lin_independent.append(row)

        # New row is linearly dependent (redundant)
        else:
            redundant.append(row)

        rank = newrank

    lin_independent = np.array(lin_independent)

    # Destructure lin_independent into equality_constraints and h
    equality_constraints = np.array(lin_independent[:, :-1])
    h = np.array(lin_independent[:, -1])
    h = np.reshape(h, newshape=(-1, 1))

    # Add zeros for slack variables
    equality_constraints = np.hstack((equality_constraints, np.zeros((equality_constraints.shape[0], slack))))

    return equality_constraints, h

# Read input, given a file pointer fp
def read_input(fp):

    g = []    # Constraint coefficients
    c = []    # Unit cost vector
    h = []    # RHS of constraints
    d = 0     # Dimension of vector space
    m = 0     # Number of constraints
    slack = 0 # Number of slack variables 

    s = f.readline().split(",")
    d = int(s[1])
    c = getlineip(f.readline().split(","))
    if s[0]=='MAX':
        c = [-x for x in c]

    todo = s[0] # "MIN" or "MAX"

    newd = d    # New dimension after adding slack variables
    slacks = {} # To store slack variables

    equality_indices = [] # Indices of rows which are equality constraints

    # Read input
    while(True):
        s = f.readline().split(",")
        if(s==[] or s==['']):
            break
        h.append(int(s[0]))

        row = []

        op = s[1]
        if op == '=':
            row = getlineip(s[2:])
            equality_indices.append(m)
        elif op == ">=":
            row = getlineip(s[2:])
            slacks[m] = newd
            newd += 1
        else:
            row = getlineip(s[2:])
            row = [-x for x in row]
            h[-1] = -h[-1]
            slacks[m] = newd
            newd += 1

        g.append(row)
        m = m+1

    # Add slack variables in constraints. Also make the vector h positive
    for i in range(len(g)):
        row = g[i]
        if d < newd:
            t = [0 for _ in range(newd - d)]
            row += t
            if(i in slacks):
                row[slacks[i]] = 1

        if h[i] < 0:
            g[i] = [-x for x in row]
            h[i] = -h[i]
    
    c += [0 for _ in range(newd - d)]

    slack = newd - d
    d = newd

    g = np.array(g)
    c = np.array(c)
    h = np.array(h)

    equality_constraints = []
    equality_rhs = []

    # Find equality constraints
    for i in equality_indices:
        equality_constraints.append(g[i, :])
        equality_rhs.append(h[i])

    # Remove redundant equality constraints
    equality_constraints, equality_rhs = remove_redundant(np.array(equality_constraints), np.array(equality_rhs), d, slack)
    eq = np.hstack((equality_constraints, equality_rhs))

    # After removal of redundant constraints
    g_prime = []
    h_prime = []

    for i in range(0, m):

        if i not in equality_indices:
            g_prime.append(g[i, :])
            h_prime.append(h[i])

        else:

            row = np.concatenate((g[i, :], [h[i]]))

            if row in eq:
                g_prime.append(g[i, :])
                h_prime.append(h[i])

    g = np.array(g_prime)
    h = np.array(h_prime)
    m = g.shape[0]

    B = [] # Linearly indepenedent columns of g
    D = [] # Rest of the columns of g
    c1 = [] # Coefficients of c corresponding to basic variables
    c2 = [] # Rest of the coefficients of c
    idx1 = [] # Indices of columns forming B
    idx2 = [] # Indices of columns forming D
    gtrans = np.transpose(g) 
    temp  = [] 
    rank = 0
    for i in range(len(gtrans)):
        row = gtrans[i]
        temp.append(row)
        newrank = np.linalg.matrix_rank(temp) # Find rank of matrix after appending new column

        # Column is linearly independent
        if(newrank==1+rank):
            B.append(row)
            c1.append(c[i])
            idx1.append(i)

        # Column is linearly dependent
        else :
            D.append(row)
            c2.append(c[i])
            idx2.append(i)

        rank=newrank

    if len(D) > 0:
        g = np.concatenate((B,D), axis=0)
    
    else:
        g = B
    
    g = np.transpose(g)                 # Coefficients of constraints (along with slack variables)
    c = np.concatenate((c1, c2))        # Unit cost vector
    idx = np.concatenate((idx1, idx2))  # To map columns to variables

    return (todo, idx, g, c, h, d, m, slack)

# Solves the LP problem using simplex algorithm
def simplex(g_rref, xB, rcc, bfs_cost, m):

    xB = np.reshape(xB, newshape=(m, 1)) # Basis variable values

    # Create initial tableau
    tableau = np.hstack((g_rref, xB))
    down = np.concatenate((np.zeros(m), rcc, [-bfs_cost]))
    tableau = np.vstack((tableau, down))

    # Perform pivot operation
    while(True):

        # Step 1: Find pivot column (q)
        found = False
        q = -1
        for i in range(tableau.shape[1]-1):
            # print(i)
            if(tableau[-1][i]<0):
                found = True
                q=i
                break
        if(not found):
            return("PASS",tableau)
        
        # Step 2: Find pivot row (p)
        found = False
        p = -1
        mn = inf
        for i in range(tableau.shape[0]-1):
            if(tableau[i][q]>0):
                found = True
                ratio = tableau[i][-1]/tableau[i][q]
                if(ratio<mn):
                    mn = ratio
                    p=i
        if(not found):
            return("UNBOUNDED",None)
        

        # Step 3: Perform pivot
        pivot = tableau[p][q]
        tableau[p, :] = tableau[p, :] / pivot
        for i in range(tableau.shape[0]):
            if i != p:
                tableau[i, :] = tableau[i, :] - tableau[i, q] * tableau[p, :]

# Find whether a column of a matrix is a column of an identity matrix
def isGood(col):

    ones = 0   # Number of ones in the column
    zeros = 0  # Number of zeros in the column
    idx = None # Index of one in column
    l = len(col)

    for i in range(l):

        if col[i] == 0:
            zeros += 1

        elif col[i] == 1:
            ones += 1
            idx = i 

        else:
            return None

    # Column is part of identity matrix. Return the index of the one
    if zeros == l - 1 and ones == 1:
        return idx

    return None

# Rearrange columns of g in such a way that the starting columns form an identity matrix 
def rearrange(g, c, idx, d):

    included = [] # Columns included in identity matrix

    for i in range(d):

        res = isGood(g[:, i]) # Find whether column can be part of identity matrix

        # continue, if column already included
        if res in included:
            continue

        if res is not None:

            included.append(res)

            col1 = np.copy(g[:, i])
            col2 = np.copy(g[:, res])

            # Swap columns i and res
            for j in range(len(col1)):
                g[j][i] = col2[j]
                g[j][res] = col1[j]

            # Swap the corresponding entries of c and idx
            c[i], c[res] = c[res], c[i]
            idx[i], idx[res] = idx[res], idx[i]

if __name__ == "__main__":

    f=open("input.csv","r")
    t = 1
    while(t>0):
        
        # Take input of the LP problem
        (todo, idx, g, c, h, d, m, slack) = read_input(f)

        B = g[:, :m] # First m columns of g
        B_inv = np.linalg.pinv(B) # Inverse of B

        xB = np.matmul(B_inv, h) # Initial BFS
        artificial = False # Whether artificial LP problem was required or not

        for x in xB:

            # BFS contains a negative variable
            if x < 0:

                # Form the matrices of artificial LP problem
                g_prime = np.concatenate((np.eye(N=m, M=m), g), axis=1)
                h_prime = np.copy(h)
                d_prime = d + m
                bfs_prime = np.concatenate((h, np.zeros(d)))

                cD_transpose = np.zeros(d)
                cB_transpose = np.ones(m)
                rcc = cD_transpose - (cB_transpose @ g)

                # Solve artificial LP problem
                ret = simplex(g_prime, h_prime, rcc, sum(h_prime), m)

                if ret[0] == "PASS":
                    
                    tableau = ret[1]

                    # Infeasible solution
                    if (tableau[-1][-1] > 1e-8) or (tableau[-1][-1] < -1e-8): 
                        print("INFEASIBLE")
                        exit()

                    # Extract information of original problem from tableau
                    g = tableau[:-1, m:-1]
                    h = tableau[:-1, -1]
                    rearrange(g, c, idx, d)

                    B = g[:, :m]
                    B_inv = np.linalg.pinv(B)

                # Unbounded LP
                elif ret[0] == "UNBOUNDED":
                    print("UNBOUNDED")
                    exit()

                artificial = True
                break

        D = g[:, m:]

        # Transform all matrices in order to form appropriate tableau 

        if not artificial:

            g = np.hstack((np.eye(N=m, M=m), B_inv @ D))
            h = B_inv @ h

        B_inv_D = g[:, m:]
        cD_transpose = c[m:]
        cB_transpose = c[:m]

        rcc = cD_transpose - (cB_transpose @ B_inv_D)
        bfs_cost = cB_transpose @ h

        # Solve LP problem
        ret = simplex(g, h, np.transpose(rcc), bfs_cost, m)

        if(ret[0] == "PASS"):

            # Map values to variables. Also, don't print values of slack variables in final solution

            g = ret[1][:m, :d]
            rearrange(g, c, idx, d)
            optimum_cost = -ret[1][-1, -1]
            bfs_opt = ret[1][:m, -1]
            sol_with_slack = np.zeros(d)

            idx = idx.astype("int32")

            for i in range(m):
                sol_with_slack[idx[i]] = bfs_opt[i]

            d -= slack
            sol = {}

            for i in range(d):

                if sol_with_slack[i] > 0:

                    if sol_with_slack[i] - int(sol_with_slack[i]) < 1e-7:
                        sol_with_slack[i] = int(sol_with_slack[i])

                    sol[i + 1] = sol_with_slack[i]

            # Check whether original problem was a maximization problem
            if todo == "MAX":
                optimum_cost *= -1

            print("PASS\n")
            print(f"Optimum cost: {optimum_cost}")
            print("Non-zero variables giving optimum cost (Index: Value):")
            print(sol)

            print("\nRedundant Constraints:")

            if len(redundant) == 0:
                print("None")
            
            for constraint in redundant:

                print(f"{constraint[-1]} = ", end="")

                for var in constraint[:-1]:
                    print(f"{var} ", end="")

                print()

        # Unbounded LP
        elif (ret[0]=="UNBOUNDED"):
            print(ret[0])
        
        t = t-1