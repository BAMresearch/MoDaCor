Instrument modules structure explainer:
--

Welcome to the location for instrument-specfific modules. If neither the base modules or the technique-specific modules satisfy the needs for data from a given instrument, you can write sprcific modules that are purpose-built and put them in here.

The subdirectory structure to place these in is recommended to follow the following convention:
./institute_abbreviation/instrument_name/module_name.

Please follow the code practices of the project, and consider making your modules available to the wider world. This will be considered if modules have the standard header, use ProcessStepDescriber to describe the module, and have tests in the corresponding directory in src/modacor/tests/modules/instrument_modules/...
