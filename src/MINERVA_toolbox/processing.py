# import pyFAI


# print("Using pyFAI verison: ", pyFAI.version)


import numpy as np
from lmfit.models import (
    GaussianModel,
    LinearModel,
    LorentzianModel,
    PseudoVoigtModel,
    SkewedGaussianModel,
    SplitLorentzianModel,
)


def parse_peak_kwargs(peakinfo, prefix):
    outparams = []
    for k, v in peakinfo.items():
        outparams.append([f"{prefix}{k}", v[0], v[1], v[2]])
    return outparams


# peak2 = fit_models[fit_type](prefix='p2_')
def fit_peaks(peaklist, x, y):

    fit_models = {
        "pvoigt": PseudoVoigtModel,
        "gaussian": GaussianModel,
        "lorentzian": LorentzianModel,
        "split_lorentzian": SplitLorentzianModel,
        "skewed_gaussian": SkewedGaussianModel,
    }
    background = LinearModel(prefix="bkg_")
    mod = background
    par_settings = []
    for ind, peak in enumerate(peaklist):
        peakprefix = f"p{ind + 1}_"
        mod += fit_models[peak["type"]](prefix=peakprefix)
        par_settings += parse_peak_kwargs(peak["settings"], peakprefix)

    pars = mod.make_params()

    pars["bkg_intercept"].set(y.min())
    pars["bkg_slope"].set(10)
    for par in par_settings:
        pars[par[0]].set(par[1], min=par[2], max=par[3])
    result = mod.fit(y, pars, x=x)
    xnew = np.arange(x.min(), x.max(), 0.001)
    comps = result.eval_components(x=xnew)
    y_fit = result.best_fit

    return result, comps, y_fit, xnew


# def import_poni(ponifile):
#     global ai
#     ai = pyFAI.load(ponifile)


# def update_pg_I07(
#     incident_angle=0.075,
#     maskfile="",
#     mask=True,
#     sample_orientation=3,
#     tilt=False,
#     tilt_angle=0,
# ):
#     global pg
#     pg = pygix.Transform()

#     pg.load(ai)
#     if mask == True:
#         pg.maskfile = maskfile
#     pg.incident_angle = incident_angle
#     pg.sample_orientation = sample_orientation
#     if tilt == True:
#         pg.tilt_angle = tilt_angle


# def integrate1d(img, qbins, qmin, qmax, chi_pos, chi_width):
#     """
#     Integrate 2D image into 1D profile over a chi sector using pygix `pg.profile_sector`.
#     """
#     intensity, q = pg.profile_sector(
#         img,
#         qbins,
#         correctSolidAngle=True,
#         method="splitpix",
#         radial_range=(qmin, qmax),
#         chi_pos=chi_pos,
#         chi_width=chi_width,
#         unit="q_A^-1",
#     )
#     return intensity, q


# def chiintegrate1d(
#     img, radial_pos, radial_width, num_points=180, chi_min=-90, chi_max=90
# ):
#     """
#     Integrate 2D image into 1D chi profile over a radial sector using pygix `pg.profile_chi`.
#     """
#     intensity, chi = pg.profile_chi(
#         img,
#         npt=num_points,
#         correctSolidAngle=True,
#         method="splitBBox",
#         radial_pos=radial_pos,
#         radial_width=radial_width,
#         chi_range=(chi_min, chi_max),
#         unit="q_A^-1",
#     )
#     return intensity, chi


# def pygix_transform_global(
#     img,
#     qbins,
#     default_range=False,
#     qxy_min=0,
#     qxy_max=0,
#     qz_min=0,
#     qz_max=0,
#     polarization_factor=0,
# ):

#     if default_range:
#         intensity, qxy, qz = pg.transform_reciprocal(
#             img,
#             npt=(qbins, qbins),
#             polarization_factor=polarization_factor,
#             method="nearest",
#             unit="A",
#         )
#     else:
#         intensity, qxy, qz = pg.transform_reciprocal(
#             img,
#             npt=(qbins, qbins),
#             ip_range=(qxy_min, qxy_max),
#             op_range=(qz_min, qz_max),
#             polarization_factor=polarization_factor,
#             correctSolidAngle=True,
#             method="nearest",
#             unit="A",
#         )

#     return intensity, qxy, qz
