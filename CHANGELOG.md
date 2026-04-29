# CHANGELOG

## v1.1.1 (2026-04-29)

### Bug fixes

* fix: better error messages for loaders ([`b643250`](https://github.com/BAMresearch/MoDaCor/commit/b6432501feaf2b2f3fefab7e2a2f809bb4f09642))

### Continuous integration

* ci: Try to publish build results only for new version, fails otherwise on pypi ([`34987d7`](https://github.com/BAMresearch/MoDaCor/commit/34987d733767c6e1d480e770109380f569f75953))

* ci: remove extra debug output from release job ([`fb5eae1`](https://github.com/BAMresearch/MoDaCor/commit/fb5eae13482693b45dd0c47b21e6698324adcecd))

* ci: move docs permissions to job-level ([`9bc3ff1`](https://github.com/BAMresearch/MoDaCor/commit/9bc3ff1afe07b28147e2c20d31f91b548432a7c8))

* ci: set permissions for deploying docs accordingly; missing version bump ([`4afa0cb`](https://github.com/BAMresearch/MoDaCor/commit/4afa0cba4ff6634b1333f9411977a3af2e1af1fd))

* ci: bump GH Actions versions in use ([`85278b7`](https://github.com/BAMresearch/MoDaCor/commit/85278b7f28053be6651b1fd93bdd8ebf0e46eedc))

* ci: syntax fix ([`189ef35`](https://github.com/BAMresearch/MoDaCor/commit/189ef35fcffdadd0c6e42c9911770a410de1e5dd))

* ci: docs info outputs in one step, declare accordingly ([`604992f`](https://github.com/BAMresearch/MoDaCor/commit/604992f4e6550dcf2958478fff6eddc3c9b3b877))

* ci: make sure GH Action step output is referencable ([`4f8be78`](https://github.com/BAMresearch/MoDaCor/commit/4f8be783cefe9b85a51d9e0ed6c485b4d3be7abf))

* ci: replace hardcoded docurl+reportpath by parsing pyproject.toml ([`a653e08`](https://github.com/BAMresearch/MoDaCor/commit/a653e08de46432b92052993c3190c4c449614c56))

* ci: deploy docs from main branch only (2) ([`b7a34f6`](https://github.com/BAMresearch/MoDaCor/commit/b7a34f6681b57018027cf58390ece95ee5e442dd))

* ci: deploy docs from main branch only ([`823fe08`](https://github.com/BAMresearch/MoDaCor/commit/823fe08f3d150dea374512e01c12e0f2a7779bb4))

* ci: prepare docs does not hurt on any branch ([`54eade5`](https://github.com/BAMresearch/MoDaCor/commit/54eade538faf825b6d3d9e632b7401c6837ab954))

* ci: publish docs via GH Actions only ([`747f484`](https://github.com/BAMresearch/MoDaCor/commit/747f4842c415d256d9ee0c7ee8b7fa3c87279061))

* ci: release job should run on main branch, no push will be made since no changes in history on PR merge expected ([`bb3cc74`](https://github.com/BAMresearch/MoDaCor/commit/bb3cc7426129ef0937b68a9d9eaa0102519fdd95))

* ci: run release job on push events only ([`b7339fe`](https://github.com/BAMresearch/MoDaCor/commit/b7339fec51a75e9dff7ddff1c7318214895918ee))

## v1.1.0 (2026-03-20)

### Bug fixes

* fix: dev environment fix ([`b8ef202`](https://github.com/BAMresearch/MoDaCor/commit/b8ef202a2233e0d3d91d566cb8319879b8e473a2))

### Code style

* style: clearer separation between functionalities, and cleanup of complexity ([`0337a08`](https://github.com/BAMresearch/MoDaCor/commit/0337a08388da0ac03c391c5ca84cda36e20f10a0))

* **Project**: remove unused (+commented) dependencies ([`655d0a9`](https://github.com/BAMresearch/MoDaCor/commit/655d0a9622629367ddca42d0440ccf84a6d8d7fe))

* **docs**: sorted imports in docs/conf.py ([`e8a80dd`](https://github.com/BAMresearch/MoDaCor/commit/e8a80dd1586373dc7c5d9363d9fa703b373b7668))

### Continuous integration

* ci: fix pyproject sem-rel cfg syntax, pt.2 ([`eee9bc3`](https://github.com/BAMresearch/MoDaCor/commit/eee9bc3937b841d388c1a28c8f464fa2e7113d30))

* ci: fix pyproject sem-rel cfg syntax ([`6400759`](https://github.com/BAMresearch/MoDaCor/commit/640075942ff7c32cc48f594657b987343232c854))

* ci: debug sem-rel not matching branch with wildcard option ([`3175bb2`](https://github.com/BAMresearch/MoDaCor/commit/3175bb2ff60827c1f8df2a278305e0a875911c8e))

* ci: pyproject syntax fix ([`34a055d`](https://github.com/BAMresearch/MoDaCor/commit/34a055dde24b74b72ad7b8f93949399df9136808))

* ci: release fails if a version number could not be determined ([`a38bbfc`](https://github.com/BAMresearch/MoDaCor/commit/a38bbfc7888c6f2dc628c0414b4973d918cd19e2))

* ci: make sure release job checks out a branch name instead of commit ([`c0f147a`](https://github.com/BAMresearch/MoDaCor/commit/c0f147a3ad833bf84b46ca17fd545c7e1a192eca))

* ci: publish shall run on main branch only ([`4e66a9a`](https://github.com/BAMresearch/MoDaCor/commit/4e66a9a293114b9cea7f94a78547859221f22b77))

* ci: run release job on branches != main only ([`a8a36ee`](https://github.com/BAMresearch/MoDaCor/commit/a8a36ee7f747c99221dbf6f68b94060043e21e14))

* ci: typo ([`a6f7e7f`](https://github.com/BAMresearch/MoDaCor/commit/a6f7e7fb4939cfcf4a834ac857c5a4d2fa20a14e))

* ci: pypi environment restricts publishing packages from main branch only ([`cac2da8`](https://github.com/BAMresearch/MoDaCor/commit/cac2da8af1def7431417cbc161619b85be56b96d))

* ci: allow semantic-release to create a new version if every branch ([`93575f6`](https://github.com/BAMresearch/MoDaCor/commit/93575f670f800ce6eb89e2ccdf7efd98f7ce44b5))

* ci: run release job on PRs, new commit for version num should be merged to main eventually ([`c645a48`](https://github.com/BAMresearch/MoDaCor/commit/c645a488c48d7b48ee1b1f27fea30f6ab56db444))

* **PyPI**: move from test.pypi.org to pypi.org ([`8028424`](https://github.com/BAMresearch/MoDaCor/commit/8028424bd6c2c61b7dc92161b19335b8248aa650))

* **GH Action**: install dependencies from pyproject.toml ([`b023a96`](https://github.com/BAMresearch/MoDaCor/commit/b023a960f091d0ca98ddd67e8889af5e71ceea0f))

* **GH Action**: allow other branches than main to be tested ([`cc1133b`](https://github.com/BAMresearch/MoDaCor/commit/cc1133b3f6ee50c101c930bc201b794e4d9906a0))

### Documentation

* docs: updated documentation to reflect state ([`2fa15fd`](https://github.com/BAMresearch/MoDaCor/commit/2fa15fdd8fb755ee616f5403ce2a3bea0ebf778e))

* **Project**: remove obsolete files ([`d516a65`](https://github.com/BAMresearch/MoDaCor/commit/d516a653e46d2a69a08a33c2912b779464a08755))

### Enh

* enh: server exposes latest error report ([`010ed6b`](https://github.com/BAMresearch/MoDaCor/commit/010ed6b9fc7185aa3404b73e1dee02616a22ba4e))

* enh: server has readiness and health reports ([`8ff76ab`](https://github.com/BAMresearch/MoDaCor/commit/8ff76ab2c3bd21832902beb5827504d9b9794f49))

### Performance improvements

* perf: move tests outside package to make it leaner ([`e85f1fa`](https://github.com/BAMresearch/MoDaCor/commit/e85f1fa67fd26d354e583e0ba83b6cde18d69800))

### Refactoring

* **Project**: update project config with copier, replaces cookiecutter ([`17a973c`](https://github.com/BAMresearch/MoDaCor/commit/17a973cf7b1635a9b55f6eab971321daf731d297))

### Unknown Scope

* Last sample endpoint ([`1384666`](https://github.com/BAMresearch/MoDaCor/commit/13846668693864739abfee071a14272f019e7d1d))

* added a dry-run option. ([`6ca935d`](https://github.com/BAMresearch/MoDaCor/commit/6ca935d08ecf50629288b93d57175e29eae110dd))

* reprioritized the remaining upgrades ([`b211659`](https://github.com/BAMresearch/MoDaCor/commit/b21165974f9e3c5c28e5a79cafc7398bc2ac5561))

* added U3: profiles/templates backend with MOUSE/SAXSess examples ([`b6774da`](https://github.com/BAMresearch/MoDaCor/commit/b6774daef34d2e7ddac8de93d9c212ed5c81870d))

* added U2 (source patch endpoint) and U5 (better run summaries) ([`30be686`](https://github.com/BAMresearch/MoDaCor/commit/30be686adf74d2eb04de31d7a476c5ecbf62adbf))

* added quick start documentation on the web API ([`b5e7659`](https://github.com/BAMresearch/MoDaCor/commit/b5e76592f4768303721965250f4c227f754b5b3c))

* added end-to-end FastAPI tests ([`39c57de`](https://github.com/BAMresearch/MoDaCor/commit/39c57ded7f8b5521111e13e9d1bc7e35f24ea775))

* web api buildout with better partial graph refresh ([`2d1e023`](https://github.com/BAMresearch/MoDaCor/commit/2d1e0234f6dc0f4545fc77c930140f76d3978df7))

* basic web functionality wired up ([`dc32c6a`](https://github.com/BAMresearch/MoDaCor/commit/dc32c6adb3a6fb29e1ccfaca17dfd61b286402c4))

* added draft server and API using FastAPI ([`9c0a821`](https://github.com/BAMresearch/MoDaCor/commit/9c0a8211e0aaad3c81b3a3f8f394c34843e7bf75))

* option to store the entire processingdata in the output hdf5 file ([`053534b`](https://github.com/BAMresearch/MoDaCor/commit/053534be602b1d953db49716b6cf8a9ce1281d09))

* adding traceability to the output HDF5 files ([`45a69d2`](https://github.com/BAMresearch/MoDaCor/commit/45a69d29247e5ff7d2a43003afd87d268a9db69d))

* CLI and ipython entry point, and added to the documentation. ([`c20b7f4`](https://github.com/BAMresearch/MoDaCor/commit/c20b7f401c7d8de28853be100b2030a58f7df19b))

* minifixes ([`d98aac3`](https://github.com/BAMresearch/MoDaCor/commit/d98aac328b009bc38378412f013021d719ebeaed))

* tried to reduce code size by removing unnecessary casts ([`7002226`](https://github.com/BAMresearch/MoDaCor/commit/7002226e525f8ffabc878d390d1a57267058800c))

* Chore: switch off matplotlib setup by pint ([`730157e`](https://github.com/BAMresearch/MoDaCor/commit/730157e9e41e54f94fc8ad78b0e30e354d0dd59d))

* Chore: matplotlib is not a dependency ([`aa3fb35`](https://github.com/BAMresearch/MoDaCor/commit/aa3fb352a9f227ad41d8de1f12512923d9aa49a7))

* adding averaging statistics (SEM and STD) to indexed_averager for Q and Psi ([`2c31853`](https://github.com/BAMresearch/MoDaCor/commit/2c318539cbef12584165219c72f22f61cd22ef27))

* improving display of mermaid graphs with optional short_title descriptor ([`5d27dae`](https://github.com/BAMresearch/MoDaCor/commit/5d27dae4d4d191a681bc81df533afe662cbf7470))

* fixing CI-CD error and modifying .gitignore ([`d0b7e0f`](https://github.com/BAMresearch/MoDaCor/commit/d0b7e0f1e3f5d64980295f62775d0d86cb3eb077))

* adding mermaid functionatlity to docs ([`551e746`](https://github.com/BAMresearch/MoDaCor/commit/551e74676bf23de48cbb69892d4da4912e1912de))

* updating badge targets ([`1a94b67`](https://github.com/BAMresearch/MoDaCor/commit/1a94b6748c43fb28915ea215eac1de4db2ad2657))

* re-adding autosummary ([`575feb0`](https://github.com/BAMresearch/MoDaCor/commit/575feb029e9e9ef6d4cb0940c13269647fd09e29))

* thoroughly reworked documentation generation ([`3a6bb67`](https://github.com/BAMresearch/MoDaCor/commit/3a6bb67d2fad1c54c1ee1346bb1e5f5ff9913b13))

* update generate_module_doc and tests to adapt to new arguments processstepdoc ([`b41e076`](https://github.com/BAMresearch/MoDaCor/commit/b41e0765acfb8ff9d4fc5db97327603d1771fc59))

* update combine_uncertainties to new arguments documentation ([`9b02521`](https://github.com/BAMresearch/MoDaCor/commit/9b025215b217f35897455e110db8519edafd31c6))

* generate module doc not in the right branch. ([`6662da1`](https://github.com/BAMresearch/MoDaCor/commit/6662da15361a0516cd5c25507c550a05052a3d89))

* draft merge-uncertainties modules for combining BaseData uncertainties ([`f3c4c12`](https://github.com/BAMresearch/MoDaCor/commit/f3c4c1259ec995a68cc64781e5314572cd63d0a0))

* drafting a tiled IoSource ([`cde0357`](https://github.com/BAMresearch/MoDaCor/commit/cde0357a00ebc0e735900f8e6d281a2c2963e761))

* removed duplicate functionality and cleaned up processstepdescriber usage, arguments dictionary is now more comprehensive ([`8ea3585`](https://github.com/BAMresearch/MoDaCor/commit/8ea35855ce35a0d2b115ffcc9863e1540eedb589))

* documentation syntax updates ([`f093b6c`](https://github.com/BAMresearch/MoDaCor/commit/f093b6c1c9fb4f978c911ca200b1322fe690c4c8))

* draft merge-uncertainties modules for combining BaseData uncertainties ([`926b263`](https://github.com/BAMresearch/MoDaCor/commit/926b2636b34f7a5c8fd57c3e00d4d291395cd657))

* drafting a tiled IoSource ([`04443b6`](https://github.com/BAMresearch/MoDaCor/commit/04443b634974a1b6609ac16f72e11bf35b340ae5))

* removing standard header from md files, and adding a module documentation generation script. ([`a75e7f5`](https://github.com/BAMresearch/MoDaCor/commit/a75e7f5126fee11e5bd678211eb741b2186da0f6))

* change rst to markdown ([`8300563`](https://github.com/BAMresearch/MoDaCor/commit/83005633191ab835554d41bdfe811b9a0a83f49b))

* early documentation structure draft ([`8fb5ad3`](https://github.com/BAMresearch/MoDaCor/commit/8fb5ad3450adf3e96f8e350ddd2b14f485d43be8))

* adding processing_data normallizer and applying it in the correction modules ([`9e6e99b`](https://github.com/BAMresearch/MoDaCor/commit/9e6e99b19ad91b1792f6d1d62d9bfa9846e3c3e9))

* adding module template for documentation ([`935152b`](https://github.com/BAMresearch/MoDaCor/commit/935152b35075c86f69796f77ce08f44338d8db59))

* applying parameter schema to all modules ([`8850997`](https://github.com/BAMresearch/MoDaCor/commit/885099721329e4e73a86766aac89998172dc796f))

* adding argument_specs schema, better default_configuration, and fixes to validate_required_keys ([`dace233`](https://github.com/BAMresearch/MoDaCor/commit/dace2333e1b676ccdb5a23f705cd6e3c326424d3))

* simple bitwise or mask to allow adding multiple masks into the main mask (following NeXus uint32 convention) ([`04fd3d6`](https://github.com/BAMresearch/MoDaCor/commit/04fd3d65da681ec9e6d717f5b01985df497f4b42))

* test for processing data sinks ([`167d6aa`](https://github.com/BAMresearch/MoDaCor/commit/167d6aa3c5060eaa199853dc66c252aefcec81ba))

* adjust pipeline run tests to include iosinks ([`b3908d3`](https://github.com/BAMresearch/MoDaCor/commit/b3908d30c359fce141655c8d91a34be4e14e5cf0))

* append-sink tests ([`1fbd2fa`](https://github.com/BAMresearch/MoDaCor/commit/1fbd2fa6a82cc8b4c073a34597e3192aab71686e))

* updated processstep tests (iosinks support) ([`6955e66`](https://github.com/BAMresearch/MoDaCor/commit/6955e6622434493182da33003e2c6fb662919a7a))

* append-processing-step tests ([`9745688`](https://github.com/BAMresearch/MoDaCor/commit/9745688f4b53d4fcede116cd4763e7bd85796abb))

* modifying pipeline to automatically add sinks to process steps ([`dc75d1b`](https://github.com/BAMresearch/MoDaCor/commit/dc75d1be29e82bea18f9c68c91373debf47fe330))

* adding a sink-processing-data process step (like append-processing-data) ([`b73ca41`](https://github.com/BAMresearch/MoDaCor/commit/b73ca4142aff7a674987d920733851847c49fd99))

* adding an append-sink process step (like append-source) ([`ff16afd`](https://github.com/BAMresearch/MoDaCor/commit/ff16afddf1a7d657c5e1cb144e9b263145c56e87))

* adding helper methods for destination resource string processing for io_sinks ([`798adc7`](https://github.com/BAMresearch/MoDaCor/commit/798adc7ffb2d0e39523cf219a0aac8394adfea8e))

* making io_sinks and io_sources optional input arguments in process_step ([`5a1ea10`](https://github.com/BAMresearch/MoDaCor/commit/5a1ea10702eb6ce4fb04d748566ee9fe7c63b246))

* a small CSV file sink ([`e4f4574`](https://github.com/BAMresearch/MoDaCor/commit/e4f4574569f78823b666cf5c9f47a94e1931e19e))

* reorganizing to match project style ([`a0c88f8`](https://github.com/BAMresearch/MoDaCor/commit/a0c88f8a51401fd3b0fb08a3b079cbff6cae531d))

* Introducing IO Sinks ([`b530c50`](https://github.com/BAMresearch/MoDaCor/commit/b530c50ee13fa021898b56dc08700ee45f775a19))

* cleanup of commented-out text ([`0138863`](https://github.com/BAMresearch/MoDaCor/commit/01388631d2902edcebd350e60065639511ce7784))

* removing unused uncertainties requirement ([`335700f`](https://github.com/BAMresearch/MoDaCor/commit/335700fcbd1b79892ba9a7dd267f5d8efa9c93d4))

* cleanup of unused methods ([`fd3d38b`](https://github.com/BAMresearch/MoDaCor/commit/fd3d38b07c619b0479079d236247d7754c7615ed))

* adding an explainer on the instrument modules ([`67e3970`](https://github.com/BAMresearch/MoDaCor/commit/67e397002a3edaa51d2c35311ffcb9443d0e5565))

* calculating X-ray scattering geometry from pixel and sample coordinates. ([`28a6ae5`](https://github.com/BAMresearch/MoDaCor/commit/28a6ae523f8f9ed58c45de706404ff7992a08845))

* fixing pixel coordinate calculation with multi-frame inputs ([`ab21e51`](https://github.com/BAMresearch/MoDaCor/commit/ab21e51a3532086771a394dccfeb7bb3757f222a))

* cleanup ([`a6e3710`](https://github.com/BAMresearch/MoDaCor/commit/a6e3710fc5153e6f9fcad4c729d263d2a7739391))

* rethinking the pixel coordinates calculator ([`7dfb3c0`](https://github.com/BAMresearch/MoDaCor/commit/7dfb3c043faadf460cec7f9bc7ffeb466a7f13ac))

* reorganisation of modules, and start on a split geometry calculator (pixel coordinates, derived matrices calculator(q, psi, etc.)) ([`7c78c1e`](https://github.com/BAMresearch/MoDaCor/commit/7c78c1e352e04d63b7aac7c35d91f78d2d808119))

* added a to_base_units to BaseData and some more small changes ([`61c38c8`](https://github.com/BAMresearch/MoDaCor/commit/61c38c8c4a80035c43910da540f153cbce7846fa))

* fix trig units ([`652010b`](https://github.com/BAMresearch/MoDaCor/commit/652010b82f540a25a4bac51133a2561e730225a0))

* fix rank of data of twotheta output ([`d830183`](https://github.com/BAMresearch/MoDaCor/commit/d830183d90c491a889fb2224fdfcb27028b93314))

* ensure units are dimensionless before certain unary operations (log and trigionometric operations) ([`04e73e4`](https://github.com/BAMresearch/MoDaCor/commit/04e73e41269ed04f2adc7dea3f4b0207bcad58f6))

* redefine pint units of pixel and px to dimensionless detector pixels ([`2aa32db`](https://github.com/BAMresearch/MoDaCor/commit/2aa32dbaa7f3dcc31dbd8ffdb66e6df766ac9b73))

* Add summary for MoDaCor library in README ([`e48565d`](https://github.com/BAMresearch/MoDaCor/commit/e48565df58b7d5086fb8ca6884b9f93a74f8fcfa))

* Allowing units label update (e.g. for setting scaling factor units) ([`4221bdf`](https://github.com/BAMresearch/MoDaCor/commit/4221bdf78539ede7be1d205832620c46a686a5db))

* fixing unit of Omega (solid angle) ([`bd6f990`](https://github.com/BAMresearch/MoDaCor/commit/bd6f99052f077efd0445dda0fc3014829817c0f2))

* fixing unit naming to make it clearer ([`62e542f`](https://github.com/BAMresearch/MoDaCor/commit/62e542fc3d0a274ce94dee590711b5207fbd2b41))

* fixing uncertainties handling with multiple keys ([`a47fe45`](https://github.com/BAMresearch/MoDaCor/commit/a47fe453cc2cfadce50d4e24e7312921aa4b0070))

* correcting logger name ([`10a0e92`](https://github.com/BAMresearch/MoDaCor/commit/10a0e92acdc858477ce1e362e26a5651eb86ae53))

* adding calculation time per processing step ([`cbe599e`](https://github.com/BAMresearch/MoDaCor/commit/cbe599e84dc4c6c9d43ceac8063a188e44fc8620))

* adding internal docs and removing unused code ([`1d60dc2`](https://github.com/BAMresearch/MoDaCor/commit/1d60dc20d93c97a3275579f7e36b3363dac04ae7))

* small improvements ([`4b735d8`](https://github.com/BAMresearch/MoDaCor/commit/4b735d825f5b63a921dc92e70d9757f2a97680e3))

* small improvements ([`a704ff0`](https://github.com/BAMresearch/MoDaCor/commit/a704ff025de3fed0744b350bed6112d4c0f52daf))

* Integration tests for the tracing information ([`e5d5d06`](https://github.com/BAMresearch/MoDaCor/commit/e5d5d0650c0bb017c49d2f8e07ac7d437f3dd8fa))

* Introducing tracing to make troubleshooting easier. Allows tracking of changes to datasets, units, uncertainties through a processing pipeline ([`8730984`](https://github.com/BAMresearch/MoDaCor/commit/8730984b085724f3ce372082a803e0fc24c2a6ed))

* small refactor for readability in find_scale_factor_1d ([`d196636`](https://github.com/BAMresearch/MoDaCor/commit/d19663684ba46caf749df764ceb5919142baa11d))

* adding multiply to init ([`16349e0`](https://github.com/BAMresearch/MoDaCor/commit/16349e0cc9408ea205c6c576cfdf929d4ceddc40))

* tiny module for multiplying data with a scale factor ([`e8e9720`](https://github.com/BAMresearch/MoDaCor/commit/e8e9720a187ed1ab0d9f31e2653ff4e42a0794d8))

* adding a scale factor finder ([`9b366aa`](https://github.com/BAMresearch/MoDaCor/commit/9b366aa3129c9bc84303c4c10db821640b30c810))

* small bugfix ([`f41a4f8`](https://github.com/BAMresearch/MoDaCor/commit/f41a4f8d7fc2e917258608a9ab0688063bdaca43))

* using attrs in all iosources ([`0468798`](https://github.com/BAMresearch/MoDaCor/commit/0468798ce9e51cdf443a7be71896cc0a285890c7))

* renaming Loader to Source ([`7fbca17`](https://github.com/BAMresearch/MoDaCor/commit/7fbca1732470e712681740c4aa797bd46086ac4d))

* added a csv source ([`51c2815`](https://github.com/BAMresearch/MoDaCor/commit/51c281587c84336585796f6537fcf454b14b6e05))

* tests for updated xs_geometry ([`e53d5ee`](https://github.com/BAMresearch/MoDaCor/commit/e53d5ee5004350281fdc7b30cc2b86094d435dba))

* bugfix in the geometry calculation - shift half a pixel ([`aec2976`](https://github.com/BAMresearch/MoDaCor/commit/aec29761d15e7aa4f181ae34e6484920c1a86966))

* bugfix in xs_geometry ([`2a0a3a0`](https://github.com/BAMresearch/MoDaCor/commit/2a0a3a060ac68c5d043f6140604149585c6e73f9))

* fix for geometry bug ([`cb5ac1a`](https://github.com/BAMresearch/MoDaCor/commit/cb5ac1abf2487daa6726fde895388c5f52ebdf2f))

* added modules for loading (extra or all) io_sources and process_data from configuration ([`71074af`](https://github.com/BAMresearch/MoDaCor/commit/71074afac05a6c5244454a7df18e09ec5c4b5545))

* adding a solid_angle_correction which is really just a divide by a BaseData in a specific databundle.. consider abstracting if we need similar divisions ([`ff812d0`](https://github.com/BAMresearch/MoDaCor/commit/ff812d0aa4824a921604a08b9357b6976911fb0e))

* adding new modules to the __init__.py registry ([`19ab2cb`](https://github.com/BAMresearch/MoDaCor/commit/19ab2cbbae176407b6e91a902ac3b7c8ac1a8f2e))

* filename change and update on required_arguments ([`af8c46c`](https://github.com/BAMresearch/MoDaCor/commit/af8c46c6a60e937518c348a187ed49af93670ac5))

* filename change and update on required_arguments ([`6b1f0e0`](https://github.com/BAMresearch/MoDaCor/commit/6b1f0e0749d9b617b29db070a79014f20d1ea586))

* added an indexed averager as the second step in the azimuthal/radial averager. Also improved nomenclature of process_step_describer works_on -> modifies ([`1f46888`](https://github.com/BAMresearch/MoDaCor/commit/1f46888c64097535d560c35cb8dee236da7c779a))

* added a pixel indexer as the first in a two-step azimuthal/radial averager ([`6499a5f`](https://github.com/BAMresearch/MoDaCor/commit/6499a5f788c20a079d4895f7845d60a8c61f8b2c))

* added the required_arguments documentation to xs_geometry ([`700b558`](https://github.com/BAMresearch/MoDaCor/commit/700b5584818cf4041e913111f56e7b818a1f4bca))

* now with MessageHandler messages ([`d56d32b`](https://github.com/BAMresearch/MoDaCor/commit/d56d32b8b0a620380647648da3762a387d3ba522))

* forgot tests ([`1ed3568`](https://github.com/BAMresearch/MoDaCor/commit/1ed3568a4bbb697436d258e4cc044b36944aaf0c))

* combining two methods for conciseness ([`1c57f16`](https://github.com/BAMresearch/MoDaCor/commit/1c57f1642b7d4b88a12adb8b182245b7b45cbcf2))

* correcting data dimensions to [..., y, x] ([`fcc8ae2`](https://github.com/BAMresearch/MoDaCor/commit/fcc8ae27054bfe216bfe43881ad3e4bf560d8fad))

* simplifying xs_geometry ([`f4d7a67`](https://github.com/BAMresearch/MoDaCor/commit/f4d7a67cc90606b354ce55c40731340608a17c0d))

* added an indexer for basedata and associated test ([`d8a8c66`](https://github.com/BAMresearch/MoDaCor/commit/d8a8c6675c78da49a572d71712cefaf77956d09a))

* updated MessageHandler with use and test in reduce_dimensionality process step ([`e75a65a`](https://github.com/BAMresearch/MoDaCor/commit/e75a65adb7b04bbf0a64882dc1aaa26ddb142462))

* fix for non-multiplicative units handling - now raises error ([`23210ab`](https://github.com/BAMresearch/MoDaCor/commit/23210aba3270ff51b8b12f78989139af492dd110))

* simplifying _VarianceDict ([`dc721fd`](https://github.com/BAMresearch/MoDaCor/commit/dc721fd2b6d2992fc28bce15d73afd3d2f53a61c))

* validation of the 'axes' metadata shape in BaseData ([`acdf1aa`](https://github.com/BAMresearch/MoDaCor/commit/acdf1aa517260c45ec9728da672f093617dcf2a6))

* improving BaseData __repr__ ([`06aca7b`](https://github.com/BAMresearch/MoDaCor/commit/06aca7b87dd878c639238b60fa0a6a8402391aa1))

* adding a copy method to BaseData ([`2dda60a`](https://github.com/BAMresearch/MoDaCor/commit/2dda60a15511c094d7751b4700b87601996a4b0b))

* improving process_step_registry ([`abbac45`](https://github.com/BAMresearch/MoDaCor/commit/abbac45cc9f6b291b5d23c9a5c8931b62921d4c5))

* fixing units case of 1d data ([`e3726ac`](https://github.com/BAMresearch/MoDaCor/commit/e3726ac450a3034c6e533ac5bd90d9bf12786dba))

* updating reduce_dimensionality to also maintain all basedata information ([`db4092f`](https://github.com/BAMresearch/MoDaCor/commit/db4092f4761d656223bbc3cba5327a6c4abf7e63))

* fixing basedata to maintain auxiliary properties in math operations ([`3f6c7de`](https://github.com/BAMresearch/MoDaCor/commit/3f6c7deea5c397f042805fea43d3fdffa7e6bec1))

* small fixes to the XS Geometry class ([`9b21584`](https://github.com/BAMresearch/MoDaCor/commit/9b215842ca7446eedbc3b64d22f762719c419f4a))

* adding calculators for X-ray scattering geometry, which add Q, Q0, Q1, Q2, Theta (scattering angle), Psi (azimuthal angle) and Omega (solid angle) ([`d98b7f8`](https://github.com/BAMresearch/MoDaCor/commit/d98b7f8ba6fa01bfd0f0060ad08dca926838a9ab))

* adding _prepared_data to ProcessStep that does not get reset after run completion ([`b24e967`](https://github.com/BAMresearch/MoDaCor/commit/b24e96730f7faff7e0f5b118ea28e2f7fde10d40))

* removed auto-uncertainties requirement ([`8694f73`](https://github.com/BAMresearch/MoDaCor/commit/8694f73a26c168303e8ef7bef3a59fa9acfa97b0))

* updated edit dates ([`5a3b13b`](https://github.com/BAMresearch/MoDaCor/commit/5a3b13b5920486477bcb68f4369f14b530e668a5))

* added to the pipeline the option of round-tripping yaml -> spec -> visual web editor (e.g. React) -> spec -> yaml ([`fa0a8f5`](https://github.com/BAMresearch/MoDaCor/commit/fa0a8f5fe55c143e1630519e442ce3b1573a2e21))

* adding support for dot/graphviz and mermaid (markdown) flowchart output of the pipeline ([`5e96306`](https://github.com/BAMresearch/MoDaCor/commit/5e9630690241d06b4268eee1aefb92fd7e7a0c4d))

* making administrative updates to ensure all files have the correct headers ([`d548a06`](https://github.com/BAMresearch/MoDaCor/commit/d548a06a0df99e3fb3550acb282158c2472cbc8f))

* adding a process step registry with tests ([`86c2683`](https://github.com/BAMresearch/MoDaCor/commit/86c2683b713e8f9736a41bbc4638d0abd75c45db))

* Sorting pipeline by ID (to avoid overwriting identical keys), allowing step_id to be string or int, and adding tests. ([`95dd12c`](https://github.com/BAMresearch/MoDaCor/commit/95dd12cb0c02b48dd0af239142512c0e6f6d4041))

* adding some internal tests ([`9daae76`](https://github.com/BAMresearch/MoDaCor/commit/9daae760593a24752cfd0d757f2ba7fa8fa02c5c))

* small fixes to pipeline ([`da24b7d`](https://github.com/BAMresearch/MoDaCor/commit/da24b7d12359cfdf78ce81b6d040d6635b5f04a2))

* Generic fixes to pipeline ([`d82bada`](https://github.com/BAMresearch/MoDaCor/commit/d82bada0f86d0d77e4ae85087a0e9db41f3cd5d2))

* expanded test for divide. working on pipeline now ([`a239c39`](https://github.com/BAMresearch/MoDaCor/commit/a239c393474fe2fea1c871d98d5c94b804478ab2))

* add pipeline integration test ([`c4fb93f`](https://github.com/BAMresearch/MoDaCor/commit/c4fb93f16c7f000037eb8746d7ff780d8475e67b))

* fix pipeline test: using the correct keyword ([`091c273`](https://github.com/BAMresearch/MoDaCor/commit/091c27341bba2f26b2997e8fe20ecb31852aedf9))

* fix pipeline test - config keys actually checked now ([`b9c5240`](https://github.com/BAMresearch/MoDaCor/commit/b9c5240bb70989d2520a49b08daa995cd3ccb95f))

* pipeline: import MoDaCor module by its name in the config ([`8ed5a9c`](https://github.com/BAMresearch/MoDaCor/commit/8ed5a9cafb5c3934f248c9a9676bc8d37d66d73e))

* renaming to improve usability ([`453a15a`](https://github.com/BAMresearch/MoDaCor/commit/453a15a41265d3406c1b2e8b4580c983ed5c2e26))

* added test cases for the basic math operations ([`03a6fa1`](https://github.com/BAMresearch/MoDaCor/commit/03a6fa1d3b8936c5ea6f049547135561d9c1fba2))

* reduce_dim_weighted can average or sum, weighted (default) or unweighted ([`0b713a0`](https://github.com/BAMresearch/MoDaCor/commit/0b713a065dbd9c9dddfc41b81e27f248800c5bc0))

* added a weighted averaging operation for reducing dimensionality ([`151093d`](https://github.com/BAMresearch/MoDaCor/commit/151093d4be822ad53d1b489ab137c9c4ac010c52))

* cleanup of maths ([`174004c`](https://github.com/BAMresearch/MoDaCor/commit/174004c9574d9fb4361bc43d35d334cce4ff0fd2))

* added maths to basedata. might split sections off later for manageability. ([`d3c1cd1`](https://github.com/BAMresearch/MoDaCor/commit/d3c1cd1ea045009e061c754e4b6cf8e98ef19404))

* extended test for poisson_uncertainties ([`9b66591`](https://github.com/BAMresearch/MoDaCor/commit/9b665919fbc05882dfbfe076c397ca3cd06f3c91))

* add test for processstep __call__ ([`4e69144`](https://github.com/BAMresearch/MoDaCor/commit/4e69144b143e361a6aeeefa3f2bda3f6190238c2))

* added a __call__ method to ProcessStep that refers to .execute() ([`bc276ca`](https://github.com/BAMresearch/MoDaCor/commit/bc276ca88aaf2c5bed3fe3569f0f087f9a79a91b))

* add configuration to ProcessSteps (in Yaml loader) ([`0739750`](https://github.com/BAMresearch/MoDaCor/commit/0739750402d1bb484c3b23d386c55f39a75d9a77))

* change call signature of pipeline: graph as first argument ([`5fe8245`](https://github.com/BAMresearch/MoDaCor/commit/5fe8245cf0b6157f35d68562019307b66580536f))

* added auto_uncertainties to the base packages, a fast uncertainties propagator with units ([`c509713`](https://github.com/BAMresearch/MoDaCor/commit/c509713123c7442cb4e385ebd1d3ee3ac09fa517))

* subtract_databundles can be used for background subtraction ([`3be9ed5`](https://github.com/BAMresearch/MoDaCor/commit/3be9ed58b9d22e53d795914567e9ecadf72eb239))

* added basic operations between DataBundles and IoSource elements ([`f5a4ae4`](https://github.com/BAMresearch/MoDaCor/commit/f5a4ae4e7c91018ad1a57000016f3f8d613532d8))

* remove unused imports ([`899e31f`](https://github.com/BAMresearch/MoDaCor/commit/899e31f2616bc435db2b9e310cf7caa784e80f0a))

* tests match modifications to modify_config_* methods ([`8ffd470`](https://github.com/BAMresearch/MoDaCor/commit/8ffd4702d61095ac9afa80a06e07c275fb201f98))

* process_step now adds default keys to self.configuration, and has more specific modify_config* methods ([`11fa741`](https://github.com/BAMresearch/MoDaCor/commit/11fa74184bee2e72e44a3c60c48949d08aefeed2))

* helper added to construct basedata from iosources location keys ([`e5c29c2`](https://github.com/BAMresearch/MoDaCor/commit/e5c29c270a138d8b07583cbe83fe2a6f73334502))

* modify_config now takes a dict or kwargs ([`e5a9745`](https://github.com/BAMresearch/MoDaCor/commit/e5a97455797148328174db4bb5d502abdfb1f5c6))

* adding the ability to extract an attribute value using get_static_metadata ([`26b4e8f`](https://github.com/BAMresearch/MoDaCor/commit/26b4e8f3fcf3e73e4c030d930897ee083d50f97b))

* adding basic operations - WIP ([`48bfc17`](https://github.com/BAMresearch/MoDaCor/commit/48bfc1722facba18b6780b8cb50fbbc34de4cee2))

* cleaning duplicate ([`ff4e399`](https://github.com/BAMresearch/MoDaCor/commit/ff4e39960e9558bf0d33f3a4bc82942def591521))

* using __repr__ instead for more convenient operation ([`21540aa`](https://github.com/BAMresearch/MoDaCor/commit/21540aa01bb369ab106504494a41c2171ae42a9a))

* adding more helpful __str__ representation in ProcessingData ([`b73f53c`](https://github.com/BAMresearch/MoDaCor/commit/b73f53c52f3497f2e99a492600df3bc63fd3abce))

* cleanup of IoRegistry ([`a07a18b`](https://github.com/BAMresearch/MoDaCor/commit/a07a18b29f7e9f565407e5a523c51dca51504949))

* minifix ([`b2e61ff`](https://github.com/BAMresearch/MoDaCor/commit/b2e61ff334e3cf343233afff9f7292ceb36443bd))

* fixed HDFLoader tests ([`2c27407`](https://github.com/BAMresearch/MoDaCor/commit/2c274075e1c9ab614c657e8f5c6d8cdabbe18382))

* automatic preload to make the IoSources more useful off the bat ([`782047a`](https://github.com/BAMresearch/MoDaCor/commit/782047ad731c9d0bbc86ad10a41d883bdd46228b))

* fix source_reference setting in IoSource instances ([`1a0b0d8`](https://github.com/BAMresearch/MoDaCor/commit/1a0b0d8dfaed0c4b4c195c99a4903486bffb49ec))

* removed outdated 'index' from iosources, done through slicing as far as I remember ([`81676c4`](https://github.com/BAMresearch/MoDaCor/commit/81676c46c163665cd37fd6b6bbd5329095a69b0f))

* addressed tests ([`b39932b`](https://github.com/BAMresearch/MoDaCor/commit/b39932b14cac3f70992d39d6abafe7927c92e94f))

* work with source's reference if needed ([`f4bec11`](https://github.com/BAMresearch/MoDaCor/commit/f4bec11067eb182c5c1ebac83db4c0ddd8795dda))

* fix header ([`a1e39f2`](https://github.com/BAMresearch/MoDaCor/commit/a1e39f225772e721722145200d8837c7aaf83f9d))

* filling out the HDFLoader and YAMLLoader methods to match IoSource ([`09206a5`](https://github.com/BAMresearch/MoDaCor/commit/09206a59baa143b95794312d151b679a36dd1507))

* better naming: resouce_location ([`17b52fd`](https://github.com/BAMresearch/MoDaCor/commit/17b52fde9d425477c3bbfa122e4c6450e78d3185))

* unifying names between iosources: YAMLLoader ([`d7d6752`](https://github.com/BAMresearch/MoDaCor/commit/d7d6752cd3d11ad35b9eed128e81d1b1473df98c))

* unifying names between iosources: HDFLoader ([`057589a`](https://github.com/BAMresearch/MoDaCor/commit/057589ad857f3b78f1cc2d52db0246126781154c))

* unifying names between iosources: YAMLLoader ([`3193454`](https://github.com/BAMresearch/MoDaCor/commit/319345462d67ebe3faa2debe55ea359c56f6f36d))

* added __all__ to YAMLLoader ([`5e34526`](https://github.com/BAMresearch/MoDaCor/commit/5e34526e4b2d052f266a1818f0a21f25c98a0dfc))

* final changes to YAMLLoader ([`537f311`](https://github.com/BAMresearch/MoDaCor/commit/537f3114f0545797b85233dd68bf120d2664d6dd))

* updating BaseData tests to deal with the simplified BaseData ([`59a4b65`](https://github.com/BAMresearch/MoDaCor/commit/59a4b65e8364bbaa7281c6e71bc55d98cb302833))

* cleaning up BaseData ([`798dc84`](https://github.com/BAMresearch/MoDaCor/commit/798dc8492448f2c5dca7014b6b4a6b68aae24876))

* simplifying BaseData, uncertainty handling should be done in processing steps ([`d9eab76`](https://github.com/BAMresearch/MoDaCor/commit/d9eab76ff9fd351631adedf3eb93e73ae0de6794))

* slight cleanup of now-simplified BaseData ([`97b38ec`](https://github.com/BAMresearch/MoDaCor/commit/97b38ec35a8e8b0a5fac06e159909eb0a065d621))

* slight cleanup of now-simplified BaseData ([`17d3bd6`](https://github.com/BAMresearch/MoDaCor/commit/17d3bd6c38d786f42474a5567a93279f59c1b837))

* define an arbitrary intensity unit ([`019a7e4`](https://github.com/BAMresearch/MoDaCor/commit/019a7e4f700d865834df0165a436b285b2a9c27a))

* Simplifying BaseData to make it workable ([`1eebce4`](https://github.com/BAMresearch/MoDaCor/commit/1eebce4384f4ec9f0b55e1cde0ce36809e702241))

* small fix ([`787df35`](https://github.com/BAMresearch/MoDaCor/commit/787df35e1a1e71423988b59d8c754bb2bc9b784b))

* added a convenience method to BaseData ([`65deec6`](https://github.com/BAMresearch/MoDaCor/commit/65deec656f592721671200c153beee50f5e714f2))

* updated basedata tests ([`6096c85`](https://github.com/BAMresearch/MoDaCor/commit/6096c8506c20b4a54be46531ba664785e1cc24f7))

* updated poisson uncertainty module tests. ([`d6aa36b`](https://github.com/BAMresearch/MoDaCor/commit/d6aa36b913a753321c8f4f177f739d4b9af28876))

* fixing tests ([`7702fbe`](https://github.com/BAMresearch/MoDaCor/commit/7702fbe3edc15979d3c1f992cc7763e6ea24f0d9))

* adjusted BaseData to match the discussed requirements ([`79b91f8`](https://github.com/BAMresearch/MoDaCor/commit/79b91f849065007ab1d4a9089f6d2a10dcc2fae0))

* applying the modifications, but in the wrong branch. will redo. ([`34a88ed`](https://github.com/BAMresearch/MoDaCor/commit/34a88edd20463b9a9002f05f60b910375848c9e5))

* there is a practical use for the to_units method ([`f6bc7d9`](https://github.com/BAMresearch/MoDaCor/commit/f6bc7d9001124f5e18ada47bfafcb4043aed5f50))

## v1.0.0 (2025-06-16)

### Unknown Scope

* updated classes diagram ([`5d3daa7`](https://github.com/BAMresearch/MoDaCor/commit/5d3daa7a3d0f078f3e4d28c1234f80926b98470d))

* extending IoSources to match IoSource for getting data attributes, shape and dtype ([`7f992ef`](https://github.com/BAMresearch/MoDaCor/commit/7f992ef625619df53720cf520da47343c789e436))

* updated iosource header ([`c179688`](https://github.com/BAMresearch/MoDaCor/commit/c179688fe1820dc7a9024e4044c953b177169891))

* updated iosource with results from 20250613 discussion ([`2dccf03`](https://github.com/BAMresearch/MoDaCor/commit/2dccf03c4b4d84256374935c3a96acd553012f5a))

* header update ([`38b8b14`](https://github.com/BAMresearch/MoDaCor/commit/38b8b1460847b9bac29c2c972c129acda6ac1db6))

* header update ([`f2a5596`](https://github.com/BAMresearch/MoDaCor/commit/f2a55967676041d3ab1a3471388c4d3922aafd66))

* header update ([`c46fe7b`](https://github.com/BAMresearch/MoDaCor/commit/c46fe7b1ead6926770c6a94f09d1e9eccd70f671))

* renaming to match hdf_loader naming ([`2afcad7`](https://github.com/BAMresearch/MoDaCor/commit/2afcad7e537b23ea58ccf7cb8829130c323a00c8))

* static_data IoSource now loads defaults from yaml with path-like interface and tests ([`99b837f`](https://github.com/BAMresearch/MoDaCor/commit/99b837fb5f53ecc781acd887a6ec8295727ab1ca))

* slowly figuring out how to realise operations ([`81c2daf`](https://github.com/BAMresearch/MoDaCor/commit/81c2daf3adee77eeb0cc4f5718dfae368e60eee5))

* making variances behave even more like a dict ([`0b8f18f`](https://github.com/BAMresearch/MoDaCor/commit/0b8f18f6e60a94ef6a4ea4714d45ad76b3b1e3ec))

* better poisson uncertainties test ([`885b6db`](https://github.com/BAMresearch/MoDaCor/commit/885b6db031bce33bf98242090911a115768b92b7))

* fixed poisson_uncertainties test ([`db0523c`](https://github.com/BAMresearch/MoDaCor/commit/db0523cdbfee5455ae2376f816dde7cc4f88dabf))

* updated basedata _VarianceDict class to make item assignment behave as expected ([`ecc26fb`](https://github.com/BAMresearch/MoDaCor/commit/ecc26fb7ecb926906b6c239fe866229653d2fa1e))

* updated basedata _VarianceDict class to make item assignment behave as expected ([`c1f5fda`](https://github.com/BAMresearch/MoDaCor/commit/c1f5fdab4212db72540fd05b423d3fff664f6502))

* fixed flake8 error ([`eef3aac`](https://github.com/BAMresearch/MoDaCor/commit/eef3aac454ebd86ea60914d49446e5098ab5609a))

* fix pytest warning: rename TestProcessingStep -> TESTProcessingStep ([`b19e701`](https://github.com/BAMresearch/MoDaCor/commit/b19e701f983322c1cc2aee68b152ad633fbb923a))

* adding to basedata test ([`32b66ed`](https://github.com/BAMresearch/MoDaCor/commit/32b66ed5b00a464f17edc39d7f040d47175d79f4))

* fixing stuff for tests ([`33185d3`](https://github.com/BAMresearch/MoDaCor/commit/33185d3e800555b6b0bb5ab7dabaee17f76c9220))

* to_units should be a processing module for clear separation of functionality ([`164579d`](https://github.com/BAMresearch/MoDaCor/commit/164579d0fcfcb786e47aaa3a83ded3407764ab95))

* updated docstring and minor fixes ([`808cd57`](https://github.com/BAMresearch/MoDaCor/commit/808cd57fbeb43352074e9157c21fbc3cbde5cdc5))

* making uncertainties (std) and variances available ([`f6ce279`](https://github.com/BAMresearch/MoDaCor/commit/f6ce2793d87f2344e1ecfad3ae029ca1d167d3f0))

* added to_units converter to BaseData ([`694c0b5`](https://github.com/BAMresearch/MoDaCor/commit/694c0b554293e5a92380a3b70f80b0278b8c451d))

* BaseData adjustment for ease-of-use and generality ([`e0276a7`](https://github.com/BAMresearch/MoDaCor/commit/e0276a7e0e00ad9d6a5c99c84669d8a77141fe00))

* adapt PoissonUncertainties unit test to use processing data ([`bf20968`](https://github.com/BAMresearch/MoDaCor/commit/bf2096899ccc0a95f58d115b3beacb37f1f00fe3))

* modification of the poisson uncertainties module for pipeline use ([`ee3e940`](https://github.com/BAMresearch/MoDaCor/commit/ee3e9400062c4d70befdb729ede685e04583cf56))

* pipeline with PoissonUncertainties step is now running ([`663fee6`](https://github.com/BAMresearch/MoDaCor/commit/663fee6313382b33ac1d00d31a08e32da0660654))

* adjust names of bundles and BaseData keys to contain "signal" ([`ff9a21a`](https://github.com/BAMresearch/MoDaCor/commit/ff9a21a4486911029284f887b320e44c14f0f270))

* begin adapting the calculate function ([`cc5f122`](https://github.com/BAMresearch/MoDaCor/commit/cc5f12217823ce3c378e15c27efcc50ec3d96c6c))

* update pipeline integration test to use new ProcessStep ([`aec63ce`](https://github.com/BAMresearch/MoDaCor/commit/aec63ce8c358448f9f3ef0f8214f3b91dd72e645))

* update to new BaseData unit handling ([`ebdceeb`](https://github.com/BAMresearch/MoDaCor/commit/ebdceebb01d251fcac65d9975a31eb968dabab71))

* use modacor's unit registry ([`5fe0eda`](https://github.com/BAMresearch/MoDaCor/commit/5fe0eda2dd0e5cf33d3b8f92155b66dbadaca61d))

* arithmetic: data is _divided_ by normalization ([`98f9389`](https://github.com/BAMresearch/MoDaCor/commit/98f938966269416c5e0280389982ed3a7297af36))

* BaseData no longer has a .data property ([`5a53260`](https://github.com/BAMresearch/MoDaCor/commit/5a53260f6cc44415b8d9b671dac8c6c6e3602a12))

* remove test of display_data property: display_units not defined ([`19d0a66`](https://github.com/BAMresearch/MoDaCor/commit/19d0a664c0074273339eb435691ae91fc1d8ddb1))

* basedata: use simplified call signature ([`619368a`](https://github.com/BAMresearch/MoDaCor/commit/619368ab05757fe6bcfd81b3c18588e551349568))

* import uncertainties.unumpy for variance calculation tests ([`7a55d68`](https://github.com/BAMresearch/MoDaCor/commit/7a55d68e7635f5c4b991778b7021d93a207647c3))

* adapt PoissonUncertainties unit test to use processing data ([`86c719b`](https://github.com/BAMresearch/MoDaCor/commit/86c719bd90a68c842e2e72118893ce099e4a1400))

* modification of the poisson uncertainties module for pipeline use ([`ce86eae`](https://github.com/BAMresearch/MoDaCor/commit/ce86eae06eee81d6353aca2f499447c3039dcaf0))

* pipeline with PoissonUncertainties step is now running ([`594315b`](https://github.com/BAMresearch/MoDaCor/commit/594315bedfdaae91e06b5afdfda85da1f4d46c62))

* adjust names of bundles and BaseData keys to contain "signal" ([`e8c342f`](https://github.com/BAMresearch/MoDaCor/commit/e8c342f6ea95d8effcca863a42a3d3d58333444c))

* add license text to license.py ([`fdd7ddf`](https://github.com/BAMresearch/MoDaCor/commit/fdd7ddf54f40de677139b9ebe61799db136ec271))

* add to last commit ([`648cb25`](https://github.com/BAMresearch/MoDaCor/commit/648cb2529665caac5d5117b7275ee99f7f069ae2))

* adding a central license to shorten headers ([`2f88d39`](https://github.com/BAMresearch/MoDaCor/commit/2f88d39c0790e2b2eb2d917f0b78493edf3c18db))

* add license text to license.py ([`0f5cfb9`](https://github.com/BAMresearch/MoDaCor/commit/0f5cfb9ceb66d7156ac01229eb46eecdaf6f3de1))

* renaming for clarity ([`da2bf41`](https://github.com/BAMresearch/MoDaCor/commit/da2bf41d1761ee99f725a1cdeadd7e63a172027d))

* small improvement to .gitignore ([`769a67d`](https://github.com/BAMresearch/MoDaCor/commit/769a67d7fd0861cfe29ef08965ec3d6b537466e1))

* example static metadata IoSource ([`555cc83`](https://github.com/BAMresearch/MoDaCor/commit/555cc83e0bbd0012dd79c6e95e11be8366e3c024))

* consistency between loaded data values and arrays, enforcing metadata ([`2655daf`](https://github.com/BAMresearch/MoDaCor/commit/2655daf8750c323df9dc030ffd76e95780f994a0))

* add to last commit ([`441068d`](https://github.com/BAMresearch/MoDaCor/commit/441068d79470cc19d13a78416dbb2a030851f2a6))

* adding a central license to shorten headers ([`939516c`](https://github.com/BAMresearch/MoDaCor/commit/939516c078bbd0d6d272399ee3c5e4094cd0718a))

* begin adapting the calculate function ([`7cf56a1`](https://github.com/BAMresearch/MoDaCor/commit/7cf56a1bdf7c96428121b37bd13ab41ded0da1bd))

* update pipeline integration test to use new ProcessStep ([`61724fd`](https://github.com/BAMresearch/MoDaCor/commit/61724fd311c1d94c86fb26a72d79f7988bc4e7e6))

* update to new BaseData unit handling ([`06f8763`](https://github.com/BAMresearch/MoDaCor/commit/06f8763c7832e8689ad9cc91977287a93b986f51))

* use modacor's unit registry ([`540ce4a`](https://github.com/BAMresearch/MoDaCor/commit/540ce4ad6ddf4ec406e515649715e8a541d856ed))

* Delete src/modacor/modules/base_modules/poisson_uncertainty.py ([`09def97`](https://github.com/BAMresearch/MoDaCor/commit/09def97ecf615beb85d130695dff425e86853d25))

* add note to self: need ProcessStep registry to populate pipeline ([`b8ba70b`](https://github.com/BAMresearch/MoDaCor/commit/b8ba70b55a1bd0eafb4c0bf39496ecc28b25ec88))

* Updating the ProcessStep definition (#34) ([`605c3f6`](https://github.com/BAMresearch/MoDaCor/commit/605c3f6151152b3dddf406b64d799bc16abc2eba))

* test and draft config format for pipeline import from yaml ([`4a40871`](https://github.com/BAMresearch/MoDaCor/commit/4a40871f2b6b17835069b376941162fa357e2481))

* initial yaml loader for pipelines ([`a12d58e`](https://github.com/BAMresearch/MoDaCor/commit/a12d58e8cd5806f742c01619d8325c4ef70d3bd7))

* add hash function to make test run (already in draft PR #33) ([`4e03a00`](https://github.com/BAMresearch/MoDaCor/commit/4e03a00765ca19b5f3ff94f3ee1a526338c2380e))

* add pyyaml to requirements for yaml definition of pipelines ([`0f76ece`](https://github.com/BAMresearch/MoDaCor/commit/0f76ecec1909671391aea861d20e0528fff688d5))

* changing unit handling to central pint unit registry ([`ee3f2f5`](https://github.com/BAMresearch/MoDaCor/commit/ee3f2f5738eecfd0068ccd2a8b729599a725fa2e))

* minor polishing ([`e9f9083`](https://github.com/BAMresearch/MoDaCor/commit/e9f9083ad87869862cabe19970b79f111db76ca3))

* adding a multiply by variable processing step ([`fa04cd7`](https://github.com/BAMresearch/MoDaCor/commit/fa04cd7f06750def211fc216cf04ea75764f29f1))

* provide azimuthal variance ([`93d0dae`](https://github.com/BAMresearch/MoDaCor/commit/93d0dae3de397e28f86e49151ef14d356e5f8c6f))

* import central unit registry ([`cfdd867`](https://github.com/BAMresearch/MoDaCor/commit/cfdd86756383f37d23bb45f5db08a7f1d1d60347))

* fix PoissonUncertainties module: needs to return outputs ([`4e4eeea`](https://github.com/BAMresearch/MoDaCor/commit/4e4eeeab00da48c34ec0bbedd78b670fcb40f07a))

* change internal_units to signal_units and removing ingest and display units ([`adfa551`](https://github.com/BAMresearch/MoDaCor/commit/adfa55185ca37b59b975ee3d978c5014bf17258b))

* use smaller array for testing ([`2b57000`](https://github.com/BAMresearch/MoDaCor/commit/2b5700015584e2267e6feaef108488241acccbc6))

* implement the test, still broken ([`2b787b7`](https://github.com/BAMresearch/MoDaCor/commit/2b787b78f13461eeffe4dab28f109ca9e0a54647))

* test runs until PoissonUncertainties are actually calculated ([`7c91102`](https://github.com/BAMresearch/MoDaCor/commit/7c91102831fdcf15c77b42a9d291d15f1e1a6d9b))

* First uncertainties commit, test is broken ([`29bec04`](https://github.com/BAMresearch/MoDaCor/commit/29bec04a256f356c89f099c5cd9c86d642e4de25))

* Starting on Poisson Testing ([`04eaa24`](https://github.com/BAMresearch/MoDaCor/commit/04eaa249dc3054bbca4990ec098d01550edb4fc2))

* prepare test running PoissonUncertainties module in a pipeline ([`537ccdf`](https://github.com/BAMresearch/MoDaCor/commit/537ccdf171d6b7c4b6549c5ba4cb8a721a69bbdf))

* adapt: DataBundle no longer has a data attribute ([`7a33677`](https://github.com/BAMresearch/MoDaCor/commit/7a33677b5ba7261a49ff1d3637d4fad45d9c0e46))

* first integration test: run pipeline with dummy processsteps ([`58e1c5d`](https://github.com/BAMresearch/MoDaCor/commit/58e1c5d2dc3fda70a02b7617f360d47c18d78ed0))

* add hash function to ProcessStep ([`15125a9`](https://github.com/BAMresearch/MoDaCor/commit/15125a96738be47231658e7dbf340cce298f0f23))

* remove DataBundle from the arguments for running the pipeline ([`def0dba`](https://github.com/BAMresearch/MoDaCor/commit/def0dbafb4ac7d6926b9d6c374033340217f0d66))

* fixing error ([`986f10b`](https://github.com/BAMresearch/MoDaCor/commit/986f10bcb6241ff3e48537c18d11fdbb389f2359))

* fixing file reference in documentation ([`5d337fd`](https://github.com/BAMresearch/MoDaCor/commit/5d337fd1c572b7c933d87a2688fd1524c11c0284))

* polish import_tester ([`683a0f8`](https://github.com/BAMresearch/MoDaCor/commit/683a0f88d51f406004f7dea7ab5c5b017cbd18b9))

* removing unused fields ([`4b12b8f`](https://github.com/BAMresearch/MoDaCor/commit/4b12b8f38b0ac2b8b30bbfeaec9ed12f6535fa5f))

* make an integrated dataset behave like a normal one ([`49e8412`](https://github.com/BAMresearch/MoDaCor/commit/49e8412e220516ab8587aaad8f9e7bb19790b45b))

* Tidy up ([`312cb82`](https://github.com/BAMresearch/MoDaCor/commit/312cb829a2c2b79e7ccf940e4e27284b98c94c86))

* HDF Loader Tests Now Passing ([`d17e05d`](https://github.com/BAMresearch/MoDaCor/commit/d17e05dfa09697d09a0d20052c6ecd5fce0bb24e))

* fix a bunch of tests ([`503d1e0`](https://github.com/BAMresearch/MoDaCor/commit/503d1e09c367bdf5a8aa8b3b18e03a25f618a242))

* adding a new poisson uncertainties estimator ([`22c2233`](https://github.com/BAMresearch/MoDaCor/commit/22c2233163903b5c613829671c261904cc7ba2dc))

* flake8 config file ([`2d34c5d`](https://github.com/BAMresearch/MoDaCor/commit/2d34c5d63743048cc49a2be485856030a5ce4ca7))

* Delete src/modacor/dataclasses/pipelinedata.py ([`5045ae9`](https://github.com/BAMresearch/MoDaCor/commit/5045ae991370e583ea5790e992ffaf5003e21d9d))

* renaming pipelinedata to processingdata ([`be76a32`](https://github.com/BAMresearch/MoDaCor/commit/be76a326c52578ce508b5740773137b1f0282e8c))

* feed-back of flake8 ([`07cdc34`](https://github.com/BAMresearch/MoDaCor/commit/07cdc34109397de6d9369d840b2e0e0326b4c636))

* Allow longer lines ([`5b74de3`](https://github.com/BAMresearch/MoDaCor/commit/5b74de3f0785a430b2b01adf1a5d8d17bce69482))

* Updated formatting with black, flake8 and isort ([`c89718b`](https://github.com/BAMresearch/MoDaCor/commit/c89718b70fd5d84d4deb6e29dc39eeb571fa37a7))

* Removing an outdated module ([`cbd7f87`](https://github.com/BAMresearch/MoDaCor/commit/cbd7f8791c98228837c8b914dab4df20e16f0b95))

* configuration for flake8 ([`dc45d13`](https://github.com/BAMresearch/MoDaCor/commit/dc45d13a44d06dc0bb3f6bf94b7a1c0cd2e7e27a))

* increase line length ([`d56fc36`](https://github.com/BAMresearch/MoDaCor/commit/d56fc36fccb1a629fb6092c1d6220b34699f7e9a))

* add __init__.py in runner submodule ([`d995aff`](https://github.com/BAMresearch/MoDaCor/commit/d995affc5210693e5e8500d64c5f2ab87c030aed))

* modify branch addition methods to operate on another Pipeline ([`3dcdce7`](https://github.com/BAMresearch/MoDaCor/commit/3dcdce7871372af6f01f84ad2466f9bf31c35f58))

* use the same syntax for incoming and outgoing branch addition ([`ee8ada7`](https://github.com/BAMresearch/MoDaCor/commit/ee8ada700b2b6850a97b0822d1c4ef8ca8f77b15))

* cleanup of databundle and adding pipelinedata ([`2015907`](https://github.com/BAMresearch/MoDaCor/commit/2015907e877b4c2e98910116a815a6a3c559a4a9))

* adding a header and cleanung up datbundle. ([`7c4e7e3`](https://github.com/BAMresearch/MoDaCor/commit/7c4e7e3888b6a3c085948cbb353ca4e37ca9342e))

* add separate methods for adding in- and outgoing branches ([`a635247`](https://github.com/BAMresearch/MoDaCor/commit/a635247014b40ef5663500adcb1f3521eadc7e2e))

* renaming raw_data to signal in BaseData ([`2f6429f`](https://github.com/BAMresearch/MoDaCor/commit/2f6429ffd65b49de93d25a218151ff3ee2e78d9d))

* fix importing BaseData ([`a104dce`](https://github.com/BAMresearch/MoDaCor/commit/a104dce54d9b2fd23932924d0cb846f7c057ce08))

* fix reference to self ([`06c2173`](https://github.com/BAMresearch/MoDaCor/commit/06c217386d3fc15bfe64bd0f15ee08bb626ec676))

* HDF Testing ([`d874571`](https://github.com/BAMresearch/MoDaCor/commit/d874571d02ffba732bbfb1ab57cf8494d8473f47))

* HDF IO Test Commit ([`920b02f`](https://github.com/BAMresearch/MoDaCor/commit/920b02f61c9e403d3d9114180a3a6bad974a8653))

* work on integration ([`125dacd`](https://github.com/BAMresearch/MoDaCor/commit/125dacdc1b3fa37e65f9c8ec6364fee380204af5))

* prepare running the pipeline - not yet tested ([`f544fea`](https://github.com/BAMresearch/MoDaCor/commit/f544fea40838c280f866e8435355d0d66e189b8f))

* Update pyproject.toml ([`7f8d1c7`](https://github.com/BAMresearch/MoDaCor/commit/7f8d1c75cd9fa949ed40364485c4f147e524935d))

* add functionality for branching the pipeline, with simple tests ([`e8093f8`](https://github.com/BAMresearch/MoDaCor/commit/e8093f802532f4a22d0222e1eb7b3618bec6ecf2))

* Update pyproject.toml ([`acf17b9`](https://github.com/BAMresearch/MoDaCor/commit/acf17b9f1196a0bf1c2b282a0bf706feec3573bd))

* Update pyproject.toml ([`70eed8a`](https://github.com/BAMresearch/MoDaCor/commit/70eed8a6560791635368186f912dace28727dbd8))

* updating the pre-commit config ([`6e247b4`](https://github.com/BAMresearch/MoDaCor/commit/6e247b412092e27ef4dec0eb00eaa4db8ee443a7))

* updated BaseData ([`72912e7`](https://github.com/BAMresearch/MoDaCor/commit/72912e78c82924ecec9a8733ec4ca8c9123c184e))

* Attempt to fix tox.ini ([`9a25376`](https://github.com/BAMresearch/MoDaCor/commit/9a253766317bd4b01ea88af1e00bca944ce957af))

* Removed unused imports ([`c5bf606`](https://github.com/BAMresearch/MoDaCor/commit/c5bf6068a18824710b29fa405988a01f2bbe9ed2))

* modifications to basedata and databundle ([`a50204c`](https://github.com/BAMresearch/MoDaCor/commit/a50204c3e620d5b9fa6ff9d038832fca22b0e623))

* updating databundle ([`f56265a`](https://github.com/BAMresearch/MoDaCor/commit/f56265af01b476c17589b46aa1028d0b238a22e6))

* Updating package metadata ([`eb4b8c8`](https://github.com/BAMresearch/MoDaCor/commit/eb4b8c8683dc4b302eb269469a580eb82cb9abf0))

* test addition of a simple hashable object ([`e9bdcbf`](https://github.com/BAMresearch/MoDaCor/commit/e9bdcbf38fabcdf75f422d857ef74ff4f55fb7f4))

* initial pipeline draft based on graphlibs TopologicalSorter ([`7c5cfc5`](https://github.com/BAMresearch/MoDaCor/commit/7c5cfc593f2dc7a38e597ba35da99b02913e55f4))

* Added tests for io sub-package ([`e0f3218`](https://github.com/BAMresearch/MoDaCor/commit/e0f321889e534b9cdf13e524878bd925766094f1))

* Modified HDF Loader ([`1fa7fd0`](https://github.com/BAMresearch/MoDaCor/commit/1fa7fd0ce72ff64851cf72625adb20045e34e575))

* Start of HDF5 Loader ([`3397b14`](https://github.com/BAMresearch/MoDaCor/commit/3397b1419343cfd35b1a9032ce8fe5e20bab4692))

* clean-up ([`5435da4`](https://github.com/BAMresearch/MoDaCor/commit/5435da48adf365f60901d718c09f0dc9e0e6e6ba))

* put back scalers are normalization factor ([`8ad59ff`](https://github.com/BAMresearch/MoDaCor/commit/8ad59ff9fe3154595dd49e23fb9fbfc209d0a648))

* Error Propagation ([`9e17100`](https://github.com/BAMresearch/MoDaCor/commit/9e1710077c2f60f1eabc196cfb6e348c6e90134c))

* Error Propagation ([`6f2854c`](https://github.com/BAMresearch/MoDaCor/commit/6f2854cb11c1ff634c88dfabf7f68209a650e96a))

* Error Propagation ([`d0b87f2`](https://github.com/BAMresearch/MoDaCor/commit/d0b87f29e0a39ff81055171aacfc48ead22bfb74))

* WIP on azimuthal integration ([`ca15a97`](https://github.com/BAMresearch/MoDaCor/commit/ca15a976fcec2ac2a31e328e552cb004b3d04a6e))

* Requirements Changes ([`d7041d5`](https://github.com/BAMresearch/MoDaCor/commit/d7041d58c667380f1f2152555d52eaef8a5599a9))

* Adding units to normalization ([`1ef17ef`](https://github.com/BAMresearch/MoDaCor/commit/1ef17ef0d6ea4c90278d25b6337f9b3b0aebe24e))

* Updates to init files ([`663ede2`](https://github.com/BAMresearch/MoDaCor/commit/663ede21782d5cc45d2b63f0a4b5bccacc84694e))

* Error Propagation ([`d0a040b`](https://github.com/BAMresearch/MoDaCor/commit/d0a040b7bcfcb6d5df8b6a6fb9b27113044e6727))

* Updated requirements ([`1c2b91c`](https://github.com/BAMresearch/MoDaCor/commit/1c2b91c7d5d0e488849c31ac6a2be50199c09d68))

* Missed comma ([`09fecf1`](https://github.com/BAMresearch/MoDaCor/commit/09fecf11165e805d3470ef0628b5ead7a0eb2137))

* Updates to the logger ([`f09b5df`](https://github.com/BAMresearch/MoDaCor/commit/f09b5df83757a281d50eb143f3c6bd86984b49f2))

* Added file headers to io and updated definitions ([`fa7996b`](https://github.com/BAMresearch/MoDaCor/commit/fa7996be2688476b861b5909eb082ec0fb25503a))

* Fixing linting ([`6bd4918`](https://github.com/BAMresearch/MoDaCor/commit/6bd4918a13ca9f9247fd80bc1cb8bc63d06670eb))

* fix renaming of ProcessStep ([`0de2352`](https://github.com/BAMresearch/MoDaCor/commit/0de2352ad4f8e2e26db1099bf0aeed28c52675a9))

* fix relative import error ([`6fa77f4`](https://github.com/BAMresearch/MoDaCor/commit/6fa77f4f518b535d224927a88c959d6f25640afb))

* dynamic requirements.txt ([`90b233b`](https://github.com/BAMresearch/MoDaCor/commit/90b233b63d250cabeb82731316d1b79df9049993))

* Populated io ([`4faa1db`](https://github.com/BAMresearch/MoDaCor/commit/4faa1dbce7c2cfdddc7356e5422a074c8311aeca))

* fix imports ([`886002c`](https://github.com/BAMresearch/MoDaCor/commit/886002c3f7f2f6e5320b56eda257e1bddbe26dc7))

* fix scatteringdata duplicate and databundle import ([`0a02cbd`](https://github.com/BAMresearch/MoDaCor/commit/0a02cbd9b119f3be445e4c8bdb173bca87ff6929))

* Tox modifications ([`b8ee1aa`](https://github.com/BAMresearch/MoDaCor/commit/b8ee1aa88e1b701785aff2ab458716137ead2913))

* Modified imports ([`7993f93`](https://github.com/BAMresearch/MoDaCor/commit/7993f938806ab4e7d526e4171557a2c01370c831))

* One step further ([`c9e0fd0`](https://github.com/BAMresearch/MoDaCor/commit/c9e0fd0e99a5a5a03a15827261c4df851343eaba))

* Modifed ProcessStep ([`22d05d1`](https://github.com/BAMresearch/MoDaCor/commit/22d05d1cbbe5080fc934e1ab668c8d59d23fde51))

* fix import all modules ([`5561826`](https://github.com/BAMresearch/MoDaCor/commit/5561826227301d80b5e61f2ff4b9a7ab8cd8fe56))

* fix all relative imports in tests ([`675e504`](https://github.com/BAMresearch/MoDaCor/commit/675e5041dfb01263c36a307b80cc557d569466e1))

* move test into project ... ([`ad171a4`](https://github.com/BAMresearch/MoDaCor/commit/ad171a43ea719c8c780eaca08f016b05191720e4))

* Changes for tox ([`142582e`](https://github.com/BAMresearch/MoDaCor/commit/142582e78d3bb5e1fc0ab6576297b9b7af66d6cc))

* databundle ([`1d18bca`](https://github.com/BAMresearch/MoDaCor/commit/1d18bca7e9a3ee2e7d8d4a649dc26bf1051738d4))

* typos ([`fb85395`](https://github.com/BAMresearch/MoDaCor/commit/fb85395f9ec5260efad6b7bc20e06e2db8ac8e3d))

* License modification ([`5d331e4`](https://github.com/BAMresearch/MoDaCor/commit/5d331e4747380a5aed4e7b0fd677a4d58ccc9dcc))

* Small updates ([`d176c49`](https://github.com/BAMresearch/MoDaCor/commit/d176c4916a27ef1c914a4085897f2b76dcc91c6c))

* migrate dask array to numpy ([`62fd3b2`](https://github.com/BAMresearch/MoDaCor/commit/62fd3b2652d53226372b152d7452b6ed261eba14))

* ScatteringData to DataBundle ([`108e8a4`](https://github.com/BAMresearch/MoDaCor/commit/108e8a406520a7c9aef0466c4df3fc133dfdc87e))

* implement base dataclass ([`8049030`](https://github.com/BAMresearch/MoDaCor/commit/80490307ddbd9e1c7bee168140139ec039e7e640))

* implement normalization and variances in dict ([`f41c0ab`](https://github.com/BAMresearch/MoDaCor/commit/f41c0ab72ade5ab0e1f7eba3b13b67fc5a7dc9c2))

* Updated processstep naming and content ([`7ae6ca7`](https://github.com/BAMresearch/MoDaCor/commit/7ae6ca7d78b7f69d3b714ee8be73ab5f5955cab0))

* adding the flow charts ([`e07cb55`](https://github.com/BAMresearch/MoDaCor/commit/e07cb55cfb1139b535e24e96b8a378bb39f5147c))

* modified to use the actual datapoint in basedata now. ([`d9bd951`](https://github.com/BAMresearch/MoDaCor/commit/d9bd9518e1cbf0829bb7df6a30d006e7a569c03d))

* Getting towards a working concept ([`0772d80`](https://github.com/BAMresearch/MoDaCor/commit/0772d80305539be57ff7323ed3360db07e998310))

* something like this ([`dfd97bf`](https://github.com/BAMresearch/MoDaCor/commit/dfd97bfdb1c6ad03ed808c36c5205e8c7f413f95))

* Getting towards a proof-of-principle ([`2a77207`](https://github.com/BAMresearch/MoDaCor/commit/2a772073dfc0445500eb27140f9a1fa55280f1a3))

* more experimentation to explore delayed execution ([`5f365a1`](https://github.com/BAMresearch/MoDaCor/commit/5f365a1c2525a826bcba4c14e3c37763f6cbf335))

* Making changes to classes based on experimentation... ([`1710f40`](https://github.com/BAMresearch/MoDaCor/commit/1710f400fd736aa8377189704fe631f50f70ac1a))

* Added some flow diagrams to help clarify... ([`b17ff22`](https://github.com/BAMresearch/MoDaCor/commit/b17ff22ae01998e94faf9e7ae9b9e7ab8919aba1))

* Update README.rst ([`1e658e2`](https://github.com/BAMresearch/MoDaCor/commit/1e658e22c527e1a3f48ca87a853e58a2bfc63130))

* Changed name of note to step_note ([`accd141`](https://github.com/BAMresearch/MoDaCor/commit/accd141a42228d1ad811bc68dfd5295b1eef73d9))

* fixed tests and validator. ([`4b768e6`](https://github.com/BAMresearch/MoDaCor/commit/4b768e6db2c5393707cf4c2d9849ca274b959585))

* Changed naming slightly, and added a message handler placeholder ([`0880270`](https://github.com/BAMresearch/MoDaCor/commit/08802703c66c9e087dbca7b35ad6c832ca7e255b))

* changes broke tests ([`e81e369`](https://github.com/BAMresearch/MoDaCor/commit/e81e3698651ca0a7fdada91a090ad85dc5c96be7))

* test implementation of basedata and processstep dataclasses ([`eb6dcc4`](https://github.com/BAMresearch/MoDaCor/commit/eb6dcc45a6aca867b78e57c8aa7d81a8b3b909da))

## v0.0.0 (2025-02-13)

### Unknown Scope

* Initial commit ([`4753f2a`](https://github.com/BAMresearch/MoDaCor/commit/4753f2a4a718cb1fbd5979f252ee90e4504866f0))
