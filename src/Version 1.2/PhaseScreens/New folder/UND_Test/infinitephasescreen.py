

from scipy import linalg
import numpy
import numba

import phasescreen, turb

__all__ = ["PhaseScreenVonKarman", "PhaseScreenKolmogorov"]


class PhaseScreen(object):

    def set_X_coords(self):
        
        self.X_coords = numpy.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = numpy.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))
        
        max_n = 1
        while True:
            if 2 ** (max_n - 1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1
        print(max_n)
        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1

            coords = numpy.round(numpy.linspace(0, self.nx_size - 1, n_points)).astype('int32')
            # print(f" for n = {n} we have {coords}")
            self.stencil[col - 1][coords] = 1
        # print(self.stencil)

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor + 1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size // 2] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)
        print(self.n_stencils)
    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = numpy.zeros((len(positions), len(positions)))

        if numba:
            calc_seperations_fast(positions, self.seperations)
        else:
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions):
                    delta_x = x2 - x1
                    delta_y = y2 - y1

                    delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

                    self.seperations[i, j] = delta_r



    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """
        self.cov_mat = turb.phase_covariance(self.seperations, self.r0, self.L0)

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
       
        
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_mat_zz.shape[0]))
        except linalg.LinAlgError:
            # print("Cholesky solve failed. Performing SVD inversion...")
            # inv_cov_zz = numpy.linalg.pinv(self.cov_mat_zz)
            raise linalg.LinAlgError("Could not invert Covariance Matrix to for A and B Matrices. Try with a larger pixel scale or smaller L0")

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)
        print(self.A_mat.shape)

    def makeBMatrix(self):
        
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = numpy.linalg.svd(BBt)
        
        L_mat = numpy.zeros((self.nx_size, self.nx_size))
        numpy.fill_diagonal(L_mat, numpy.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    def make_initial_screen(self):
        
        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        self._R = numpy.random.default_rng(self.random_seed)

        self._scrn = phasescreen.ft_phase_screen(
            self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        )

        self._scrn = self._scrn[:, :self.nx_size]
        # print(self.pixel_scale)
        # print(self.stencil_positions)
        # print(self.stencil_length)
        # print(self._scrn[0].size)

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)

        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row()

        self._scrn = numpy.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn
    
    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]

    

class PhaseScreenVonKarman(PhaseScreen):
   
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2):

        self.n_columns = n_columns

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()


    def set_stencil_coords(self):
        print("HI")
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))
        self.stencil[:self.n_columns] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)
        print(self.stencil_coords)
        print(self.stencil_positions)


def find_allowed_size(nx_size):
    
    n = 0
    while (2 ** n + 1) < nx_size:
        n += 1

    nx_size = 2 ** n + 1
    return nx_size


class PhaseScreenKolmogorov(PhaseScreen):
    
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, stencil_length_factor=4):

        self.requested_nx_size = nx_size
        self.nx_size = find_allowed_size(nx_size)
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size
        self.random_seed = random_seed

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()
        
    # Read and Understand this
    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        # print(stencil_data)
        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value

        new_row.shape = (1, self.nx_size)
        return new_row


    def __repr__(self):
        return str(self.scrn)
    


@numba.jit(nopython=True, parallel=True)
def calc_seperations_fast(positions, seperations):

    for i in numba.prange(len(positions)):
        x1, y1 = positions[i]
        for j in range(len(positions)):
            x2, y2 = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r