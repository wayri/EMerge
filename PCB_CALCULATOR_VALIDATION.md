# PCB calculator validation and limits

The `PCBCalculator` line methods accept geometry in the constructor's stackup
unit (`unit` metres per user unit). Their inverse search bounds use that same
unit. Core formula functions accept SI metres and hertz. An inverse request now
raises `ValueError` if its target cannot be reached within the supplied bounds;
it does not return the nearest boundary as though the target had been met.

The Kirschning/Jansen microstrip characteristic-impedance dispersion equation
is used only within its published range: `0.1 <= W/h <= 10`,
`1 <= er <= 18`, and `h / vacuum_wavelength <= 0.1`. A homogeneous dielectric
is required for the scalar stackup model. Mixed or absent dielectric data
requires a field solver or a separately justified explicit effective-er
override. An override is an engineering assumption, not a solver result.

These closed-form quasi-TEM calculations do not establish USB, DDR, PCIe,
Ethernet or other protocol compliance. Connector transitions, return-path
discontinuities, loss, crosstalk, eye diagrams, equalization and receiver
limits need device-specific models, measurements or a qualified field/channel
solver. The regression tests exercise unit conversion, inverse round trips,
out-of-range targets, missing materials and the dispersion coefficient.

References: [Qucs single microstrip equations](https://qucs.github.io/tech/node75.html)
and [Qucs implementation](https://github.com/Qucs/qucsator/blob/develop/src/components/microstrip/msline.cpp).
