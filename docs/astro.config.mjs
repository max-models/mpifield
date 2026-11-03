// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/mpifield/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'mpifield',
	title: 'mpifield',
	description: 'Fields on uniform grids split over MPI ranks, with compiled linear gather and scatter for particle codes.',
	pythonPackage: 'mpifield',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, from the shape functions to the field operations, and a Numerics section on each
	// method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'mpifield.kernels.linear_cell': 'Linear weights',
			'mpifield.kernels.quadratic_cell': 'Quadratic weights',
			'mpifield.weighting': 'Deposit and gather',
			'mpifield.field': 'Halos, gradient, smoothing',
			'mpifield.prescribed': 'Prescribed fields',
			'mpifield.evaluate': 'Evaluation',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/mpifield/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Examples',
			items: [
				{ autogenerate: { directory: 'examples' } },
				{ label: 'Scripts', items: [{ autogenerate: { directory: 'example-scripts' } }] },
			],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Testing', slug: 'development/testing' },
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
