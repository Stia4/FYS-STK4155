from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# TODO:
# Plot beta as function of polynomial order
# Plot the MSE/R2 results, not just print them
# Add scaling/centering of data (e.g. by the mean)
# and explanation why you might do this

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.] (design matrix)
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polinomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def train_test_data(x, y, z, i):
	"""
	Takes in x,y and z arrays, and a array with random indesies iself.
	returns learning arrays for x, y and z with (N-len(i)) dimetions
	and test data with length (len(i))
	"""
	if len(x.shape) > 1:
		x = x.flatten()
		y = y.flatten()
		z = z.flatten()

	x_learn = x[:-i]
	y_learn = y[:-i]
	z_learn = z[:-i]
	x_test  = x[-i:]
	y_test  = y[-i:]
	z_test  = z[-i:]

	return x_learn, y_learn, z_learn, x_test, y_test, z_test

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
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
x_, y_ = np.meshgrid(x, y)
z = FrankeFunction(x_, y_)
z += np.random.normal(0, 0.1*np.mean(z), z.shape) # Adding some normal noise

### Split data into training and test data, using ~25% as test data
test_size = int(0.25*x.size)
x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data(x_, y_, z, test_size)

### Make design matrix, calculate beta for OLS, and get model, orders 2 to 6 covered
order = [2, 6]
X_lrn    = dict()
X_tst    = dict()
beta_OLS = dict()
z_lrn_model  = dict()
z_tst_model  = dict()
for i in range(order[0], order[1]+1):
	X_lrn[i]    = create_X(x_lrn, y_lrn, i)
	X_tst[i]    = create_X(x_tst, y_tst, i)
	print(X_lrn[i].shape)
	beta_OLS[i] = np.linalg.pinv(X_lrn[i].T @ X_lrn[i]) @ X_lrn[i].T @ z_lrn
	z_lrn_model[i]  = np.reshape(X_lrn[i] @ beta_OLS[i], z_lrn.shape)
	z_tst_model[i]  = np.reshape(X_tst[i] @ beta_OLS[i], z_tst.shape)

### Check fit using MSE and R2, printing as a nice table
header  = "".join(["|{:^10d}".format(i) for i in beta_OLS.keys()])
MSE_lrn = "".join(["|{:^10.2e}".format(     MSE(z_lrn, z_lrn_model[i])) for i in beta_OLS.keys()])
MSE_tst = "".join(["|{:^10.2e}".format(     MSE(z_tst, z_tst_model[i])) for i in beta_OLS.keys()])
R2_lrn  = "".join(["|{:^10.2e}".format(R2_Score(z_lrn, z_lrn_model[i])) for i in beta_OLS.keys()])
R2_tst  = "".join(["|{:^10.2e}".format(R2_Score(z_tst, z_tst_model[i])) for i in beta_OLS.keys()])
print("\n lrn " + header)
print(" MSE " + MSE_lrn)
print(" MSE " + R2_lrn + "\n")
print(" tst " + header)
print(" MSE " + MSE_tst)
print(" MSE " + R2_tst + "\n")

### Make plot(s)
fig = plt.figure(figsize=plt.figaspect(1.0/3.0))
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot_surface(x_, y_, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title("Data")
for i in beta_OLS.keys():
	ax = fig.add_subplot(2, 3, i-min(beta_OLS.keys())+2, projection='3d')
	ax.plot_surface(x_, y_, np.concatenate([z_lrn_model[i], z_tst_model[i]]).reshape(z.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title(f"Pol. deg {i}")
plt.show()
