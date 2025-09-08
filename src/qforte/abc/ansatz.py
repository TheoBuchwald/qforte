"""
Ansatz base classes
====================================
The mixin classes inherited by any algorithm that uses a parameterized
ansatz. Member functions should be minimal and aim only to implement
the ansatz circut and potential supporting utility functions.
"""

import numpy as np
import qforte as qf

from qforte.utils.state_prep import build_refprep
from qforte.utils.trotterization import trotterize
from qforte.utils.compact_excitation_circuits import compact_excitation_circuit
from qforte.abc.mixin import Trotterizable


class UCC(Trotterizable):
    """A mixin class for implementing the UCC circuit ansatz, to be inherited by a
    concrete class UCC+algorithm class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ansatz_circuit(self, amplitudes=None):
        """This function returns the Circuit object built
        from the appropriate amplitudes.

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """
        temp_pool = qf.SQOpPool()
        tamps = self._tamps if amplitudes is None else amplitudes

        if self._pool_type == "sa_SD":
            # The elements of the sa_SD pool are typically linear combinations of
            # second-quantized, fermionic, anti-Hermitian excitation operators.
            # To avoid breaking particle number and Sz symmetry, a temporary
            # SQOpPool is created where each spin-orbital, anti-Hermitian,
            # second-quantized operator is treated as a single element
            # (first-order Trotterization).
            for tamp, top in zip(tamps, self._tops):
                sa_sq_op = self._pool_obj[top][1].terms()
                # The first half of the terms encode the de-excitation
                # component of the anti-Hermitian, singlet-spin adapted operator.
                half_length = len(sa_sq_op) // 2
                for coeff, cr, ann in sa_sq_op[:half_length]:
                    sq_op = qf.SQOperator()
                    sq_op.add_term(coeff, cr, ann)
                    # Add the corresponding excitation component.
                    sq_op.add_term(-coeff, ann, cr)
                    temp_pool.add(tamp, sq_op)
        else:
            for tamp, top in zip(tamps, self._tops):
                temp_pool.add(tamp, self._pool_obj[top][1])

        if self._compact_excitations:
            U = qf.Circuit()
            for tamp, sq_op in temp_pool:
                U.add(
                    compact_excitation_circuit(
                        tamp * sq_op.terms()[1][0],
                        sq_op.terms()[1][1],
                        sq_op.terms()[1][2],
                        self._qubit_excitations,
                    )
                )
            return U

        A = temp_pool.get_qubit_operator(
            "commuting_grp_lex", qubit_excitations=self._qubit_excitations
        )

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        if phase1 != 1.0 + 0.0j:
            raise ValueError(
                "Encountered phase change, phase not equal to (1.0 + 0.0i)"
            )
        return U

    def build_orb_energies(self):
        """
        This code provides the spin-orbital energies used in
        Jacobi iterations. If the orbital energies have not
        been provided by an external software (Psi4), they
        are computed internally.
        """
        self._orb_e = []
        ### WARNING: Assuming RHF orbital energies ###
        if hasattr(self._sys, "hf_orbital_energies"):
            print("\nSingle-particle energies")
            print("------------------------", flush=True)
            for i, j in enumerate(self._sys.hf_orbital_energies):
                self._orb_e += [j] * 2
                print(f" {2*i:3} {j:+16.12f}", flush=True)
                print(f" {2*i+1:3} {j:+16.12f}", flush=True)
        else:
            print("\nBuilding single-particle energies:")
            print("---------------------------------------", flush=True)
            if getattr(self, "computer", None) is not None:
                qc = qf.Computer(self.computer)
                uses_computer = True
            else:
                qc = qf.Computer(self._nqb)
                qc.apply_circuit(self._refprep)
                uses_computer = False
            E0 = qc.direct_op_exp_val(self._qb_ham)

            for i in range(self._nqb):
                if uses_computer:
                    qc = qf.Computer(self.computer)
                    ref_ = self.computer.get_refs()[0]
                    ref = [int(i) for i in bin(ref_[0])[2:]][::-1] + (self._nqb - len(bin(ref_[0])[2:])) * [0]
                else:
                    qc = qf.Computer(self._nqb)
                    qc.apply_circuit(self._refprep)
                    ref = self._ref
                qc.apply_gate(qf.gate("X", i, i))
                Ei = qc.direct_op_exp_val(self._qb_ham)

                if i < sum(ref):
                    ei = E0 - Ei
                else:
                    ei = Ei - E0

                print(f"  {i:3}     {ei:+16.12f}", flush=True)
                self._orb_e.append(ei)

    def get_res_over_mpdenom(self, residuals):
        """This function returns a vector given by the residuals divided by the
        respective Moller Plesset denominators.

        Parameters
        ----------
        residuals : list of floats
            The list of (real) floating point numbers which represent the
            residuals.
        """

        if getattr(self, "computer", None) is not None and len(self.computer.get_refs()) > 1:

            if not hasattr(self, "_jac") and not getattr(self, "_extended_jacobian", False):
                num_amp = len(self._tops)
                self._jac = np.zeros((num_amp, num_amp), dtype=np.complex128)
                for mu, m in enumerate(self._tops):
                    mu_det = self._excited_dets[m]
                    for nu, n in enumerate(self._tops):
                        jac_val = 0.0
                        nu_det = self._excited_dets[n]
                        for M, mu_coeff in mu_det.items():
                            for N, nu_coeff in nu_det.items():
                                if M != N:
                                    continue
                                jac_val -= mu_coeff * nu_coeff
                        self._jac[mu, nu] = jac_val
                for nu, n in enumerate(self._tops):
                    nu_op = self._pool_obj[n][1]
                    sq_creators = nu_op.terms()[0][1]
                    sq_annihilators = nu_op.terms()[0][2]
                    self._jac[:, nu] *= (
                        sum(self._orb_e[x] for x in sq_annihilators)
                        - sum(self._orb_e[x] for x in sq_creators)
                    )
            elif not hasattr(self, "_jac"):
                num_amp = len(self._tops)
                self._jac = np.zeros((num_amp, num_amp), dtype=np.complex128)
                for nu, n in enumerate(self._tops):
                    comm = self._commutator_pool[n][1]
                    qc_comm = qf.Computer(self.computer)
                    qc_comm.apply_operator(comm)
                    nu_coeffs = qc_comm.get_coeff_vec()
                    for mu, m in enumerate(self._tops):
                        mu_det = self._excited_dets[m]
                        jac_val = 0.0
                        for M, mu_coeff in mu_det.items():
                            jac_val -= mu_coeff * nu_coeffs[M]
                        self._jac[mu, nu] = jac_val

            remove_redundancies = getattr(self, "_remove_redundancies", True)
            if remove_redundancies:
                try:
                    resids_over_denoms = np.linalg.solve(self._jac, np.reshape(residuals, (len(residuals), 1))).flatten()
                except np.linalg.LinAlgError:
                    # If the system is singular, we use lstsq to find a least-squares solution
                    resids_over_denoms = np.linalg.lstsq(self._jac, np.reshape(residuals, (len(residuals), 1)), rcond=None)[0].flatten()
            else:
                resids_over_denoms = np.linalg.lstsq(self._jac, np.reshape(residuals, (len(residuals), 1)), rcond=None)[0].flatten()
        else:
            resids_over_denoms = []

            # loop over toperators
            for mu, m in enumerate(self._tops):
                sq_op = self._pool_obj[m][1]

                temp_idx = sq_op.terms()[0][2][-1]
                if self._ref[temp_idx]:  # if temp_idx is an occupied idx
                    sq_creators = sq_op.terms()[0][1]
                    sq_annihilators = sq_op.terms()[0][2]
                else:
                    sq_creators = sq_op.terms()[0][2]
                    sq_annihilators = sq_op.terms()[0][1]

                denom = sum(self._orb_e[x] for x in sq_annihilators) - sum(
                    self._orb_e[x] for x in sq_creators
                )

                res_mu = residuals[mu] / denom

                resids_over_denoms.append(res_mu)

        return resids_over_denoms
