from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import mean_squared_error as MSE_skl
from sklearn.metrics import r2_score as R2_skl

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams.update({'font.size': 16})
np.random.seed(0)

def FrankeFunction(x,y):
	"""
	Base function to which we fit the polynomials
	"""
	term1 =  0.75*np.exp(-(9*x-2)**2/4.00 - (9*y-2)**2/4.00)
	term2 =  0.75*np.exp(-(9*x+1)**2/49.0 - (9*y+1)   /10.0)
	term3 =  0.50*np.exp(-(9*x-7)**2/4.00 - (9*y-3)**2/4.00)
	term4 = -0.20*np.exp(-(9*x-4)**2      - (9*y-7)**2     )
	return term1 + term2 + term3 + term4

def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.] (design matrix)
	Input is x and y mesh, or raveled mesh, keyword agruments n is the degree of the polynomial
	you want to fit.
	"""
	if len(x.shape) > 1: # Unravels/flattens multidimensional arrays
		x = x.flatten()
		y = y.flatten()

	N = len(x)
	l = int((n + 1)*(n + 2)/2) # Number of elements in beta
	X = np.ones((N, l))

	for i in range(1, n + 1):
		q = int((i)*(i + 1)/2)
		for k in range(i + 1):
			X[:,q + k] = (x**(i - k))*(y**k)

	return X

def train_test_data_evenly(x, y, z, n = 4):
	"""
	Takes in x, y, and z arrays, and splits each into learn and train sets.
	Puts every n-th element into test set and rest in training set.
	E.g. n = 4 sets 1/4 of the data as test and 3/4 as training.
	"""
	if len(x.shape) > 1:
		x = x.flatten()
		y = y.flatten()
		z = z.flatten()
	
	nth = slice(None, None, n)
	x_learn = np.delete(x, nth)
	y_learn = np.delete(y, nth)
	z_learn = np.delete(z, nth)
	x_test  = x[nth]
	y_test  = y[nth]
	z_test  = z[nth]

	return x_learn, y_learn, z_learn, x_test, y_test, z_test

def combine_train_test(z_learn, z_test, n = 4):
	"""
	Recombines test and train data which has been split by train_test_data_evenly.
	"""				
	z = np.zeros(len(z_learn)+len(z_test))

	mask = np.zeros(z.shape, dtype=bool)
	mask[slice(None, None, n)] = True # test: true, train: false

	z[~mask] = z_learn
	z[ mask] = z_test

	return z

def MSE(y, y_tilde):
	"""
	Function for computing mean squared error.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return np.sum((y-y_tilde)**2)/y.size

def R2_Score(y, y_tilde):
	"""
	Function for computing the R2 score.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return 1 - np.sum((y[:-2]-y_tilde[:-2])**2)/np.sum((y[:-2]-np.average(y))**2)

### Make goal data
x = np.arange(0, 1, 0.02)
y = np.arange(0, 1, 0.02)
x_, y_ = np.meshgrid(x, y)
z = FrankeFunction(x_, y_)
z += np.random.normal(0, 0.1*np.mean(z), z.shape) # Adding some normal noise
z -= np.mean(z)

### Split data into training and test data, using ~25% as test data
test_size = 4 # takes 1/4 of the data as test data
x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x_, y_, z, test_size)

### Make design matrix, calculate beta for OLS, and get model, orders 2 to 6 covered
orders = [2, 3, 4, 5, 6]
X_lrn    = dict()
X_tst    = dict()
beta_OLS = dict()
z_lrn_model  = dict()
z_tst_model  = dict()
for i in orders:
	X_lrn[i]    = create_X(x_lrn, y_lrn, i)
	X_tst[i]    = create_X(x_tst, y_tst, i)
	beta_OLS[i] = np.linalg.pinv(X_lrn[i].T @ X_lrn[i]) @ X_lrn[i].T @ z_lrn
	z_lrn_model[i] = np.reshape(X_lrn[i] @ beta_OLS[i], z_lrn.shape)
	z_tst_model[i] = np.reshape(X_tst[i] @ beta_OLS[i], z_tst.shape)

### Plotting beta as function of order
betas = []
for n in orders:
	for i in range(int((n + 1)*(n + 2)/2)-1):
		if i >= len(betas):
			betas.append([beta_OLS[n][i]])
		else:
			betas[i].append(beta_OLS[n][i])
for beta in betas:
	if len(beta) != 1:
		plt.plot(orders[len(orders)-len(beta):], beta)
	else: # Coefficients with only one occurance are plotted as points
		plt.plot(orders[len(orders)-len(beta):], beta, "o")
plt.xticks(ticks=orders)
plt.xlabel("Polynome order")
plt.ylabel(r"Coefficient $\beta$")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("fig/Beta.pdf", format="pdf")
plt.clf()

### Check fit using MSE and R2, printing as a nice table, using both our and sklearn's MSE/R2 functions
header  = "".join(["|{:^10d}".format(i) for i in orders])
MSE_lrn = [     MSE(z_lrn, z_lrn_model[i]) for i in orders]
MSE_tst = [     MSE(z_tst, z_tst_model[i]) for i in orders]
R2_lrn  = [R2_Score(z_lrn, z_lrn_model[i]) for i in orders]
R2_tst  = [R2_Score(z_tst, z_tst_model[i]) for i in orders]
MSE_lrn_skl = [MSE_skl(z_lrn, z_lrn_model[i]) for i in orders]
MSE_tst_skl = [MSE_skl(z_tst, z_tst_model[i]) for i in orders]
R2_lrn_skl  = [ R2_skl(z_lrn, z_lrn_model[i]) for i in orders]
R2_tst_skl  = [ R2_skl(z_tst, z_tst_model[i]) for i in orders]
print("\n learn  " + header)
print("MSE     "+"".join(["|{:^10.2e}".format(MSE) for MSE in     MSE_lrn])       )
print("MSE_skl "+"".join(["|{:^10.2e}".format(MSE) for MSE in MSE_lrn_skl])       )
print("R2      "+"".join(["|{:^10.2e}".format( R2) for  R2 in      R2_lrn])       )
print("R2_skl  "+"".join(["|{:^10.2e}".format( R2) for  R2 in  R2_lrn_skl]) + "\n")
print("  test  " + header)
print("MSE     "+"".join(["|{:^10.2e}".format(MSE) for MSE in     MSE_tst])       )
print("MSE_skl "+"".join(["|{:^10.2e}".format(MSE) for MSE in MSE_tst_skl])       )
print("R2      "+"".join(["|{:^10.2e}".format( R2) for  R2 in      R2_tst])       )
print("R2_skl  "+"".join(["|{:^10.2e}".format( R2) for  R2 in  R2_tst_skl]) + "\n")

### Plotting the MSE/R2 as function of polynomial degree
plt.plot(orders, MSE_lrn, "o-", label="MSE Training data")
plt.plot(orders, MSE_tst, "o-", label="MSE Test data")
plt.hlines(0, min(orders), max(orders), colors="gray", linestyles="dashed")
plt.xticks(ticks=orders)
plt.xlabel("Polynome order")
plt.ylabel("Mean squared error")
plt.legend(loc=1)
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("fig/MSE.pdf", format="pdf")
plt.clf()
plt.plot(orders, R2_lrn, "o-", label="R2 Score Training data")
plt.plot(orders, R2_tst, "o-", label="R2 Score Test data")
plt.hlines(1, min(orders), max(orders), colors="gray", linestyles="dashed")
plt.xticks(ticks=orders)
plt.xlabel("Polynome order")
plt.ylabel("R2 Score")
plt.legend(loc=4)
plt.tight_layout()
plt.grid(alpha = 0.3)
plt.savefig("fig/R2.pdf", format="pdf")
plt.clf()

### Make plot(s) of surfaces, both data and polynomials
ratio = np.array([2, 3], dtype=int) # Grid ratio ratio[0]/ratio[1] (integers!), e.g. screen ratio 16/9 would be [9, 16]
dims = ratio * int(np.ceil(np.sqrt((len(orders)+1)/(ratio[0]*ratio[1])))) # Elements to place in x/y to best match ratio
view = [30, 90] # Viewing angle in degrees, [height, rotation]

fig = plt.figure(figsize=plt.figaspect(ratio[0]/ratio[1]))
ax = fig.add_subplot(*dims, 1, projection='3d')
ax.plot_surface(x_, y_, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xticks([0.0, 0.5, 1.0]) # Reducing amount of tickmarks for clarity
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_zticks([-0.5, 0.0, 0.5])
ax.set_title("Data", y=1.0, pad=0)
ax.view_init(*view)
for i in orders:
	ax = fig.add_subplot(*dims, i-min(orders)+2, projection='3d')
	ax.plot_surface(x_, y_, combine_train_test(z_lrn_model[i], z_tst_model[i], test_size).reshape(z.shape),
					cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xticks([0.0, 0.5, 1.0])
	ax.set_yticks([0.0, 0.5, 1.0])
	ax.set_zticks([-0.5, 0.0, 0.5])
	ax.set_title(f"Pol. deg {i}", y=1.0, pad=0)
	ax.view_init(*view)
fig.suptitle("Model applied to entire set")
fig.tight_layout()
plt.savefig("fig/Surfaces.pdf", format="pdf")
plt.clf()
