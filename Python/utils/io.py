import numpy  as np
import pandas as pd
import utils

class SMLAResultsReader(object):
	def __init__(self, path):
		self._path = path

	def __del__(self):
		self.close()

	def __enter__(self):
		self.open()
		return self

	def __exit__(self, *args):
		self.close()

	def open(self):
		store = pd.HDFStore(self._path, 'r')
		results = store['results']
		tparams = store['task_parameters']
		mparams = { k.split('/')[2]:store[k] for k in store.keys() if k.startswith('/method_parameters/')}
		meta    = store['meta']
		self._data  = (results, tparams, mparams)
		self._store = store
		self._meta  = meta

	def close(self):
		self._store.close()

	def get_smla_names(self):
		return np.array([ k for k in self.get_names() if any(self._meta.smla_names==k) ])

	def get_names(self):
		return np.array([ k.split('/')[2] for k in self._store.keys() if k.startswith('/method_parameters/') ])

	def get_col(self, k, name=None, return_filtered=False):
		results, tparams, mparams = self._data
		I = np.repeat(True, len(results))
		# print('#'*100)
		# print(name)
		# print(mparams.keys())
		# print('#'*100)
		# print(mparams.keys())
		if not(name is None) and (k in mparams[name].keys()):
			I  = (results.name == name).values
			mp = mparams[name].assign(pid=mparams[name].index, name=name)
			out = results.merge(mp, on=['name','pid'], how='left')[k]
		elif (k in tparams.keys()):
			out = results.merge(tparams, left_on='tid', right_index=True, how='left')[k]
		elif (k in results.keys()):
			out = results[k]
		else:
			out = pd.Series( np.repeat(np.nan,len(results)), results.index)
		return out[I] if return_filtered else (out, I)

	def extract(self, labels, name=None, **constraints):
		if not(name is None):
			constraints['name'] = name
		results, tparams, mparams = self._data
		has_filters = (len(constraints) > 0)
		Is = []
		df = results.loc[:,['name','tid','pid']]
		# Restrict the data based on column:value pairs in constraints
		# Also, if any columns are in labels, save those columns
		if has_filters:
			for k,v in constraints.items():
				d, I = self.get_col(k, name=name)
				if isinstance(v, (str,bool)):
					I[I] = (d[I]==v)
				else:
					if np.isinf(v):
						I[I] = np.isinf(d[I])
					else:
						J = np.isinf(d[I])
						if any(J):
							I[J] = False
							I[~J] = np.isclose(d[I[~J]], v)
						else:
							I[I] = np.isclose(d[I], v)
				Is.append(I.copy())
				if k in labels:
					df[k] = d.values
			I = np.logical_and.reduce(Is) if (len(Is) > 0) else df.index
		# Add any other columns in labels that weren't stored earlier
		for l in labels:
			if not(l in df.keys()):
				d, _ = self.get_col(l, name=name)
				df[l] = d.values
		# Return the data, applying the filter if there is one
		return df[I] if has_filters else df

	def get_mn_std(self, x, y, name=None, filter_nan_xs=True, ignore_infs=True, **constraints):
		L = [x,y] if isinstance(x,str) else x + [y]
		E = self.extract(L, name=name, **constraints)[L]
		# if filter_nan_xs:
		# 	E = E[E[x].notna()]
		E = E.astype(float)
		if ignore_infs:
			E = E[np.isfinite(E[y])]
		E[y] = E[y].astype(float)

		E_gb = E.groupby(x)[y]
		# data = { 'mean' : E_gb.mean(),
		# 	     'std'  : E_gb.std(),
		# 	     'n'    : E_gb.count() }
		D = E_gb.aggregate([ 'mean', 'std', 'count' ])
		D['x'] = D.index
		D = D.reset_index(drop=True)
		if any(E[x].isna()):
			r = E[E[x].isna()][y].aggregate([ 'mean', 'std', 'count' ])
			r['x'] = np.nan
			D = D.append(r, ignore_index=True)
		# data['x'] = data['n'].index.to_series()
		# df = pd.DataFrame(data).reset_index(drop=True)
		return D
