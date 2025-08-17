# Interface Conformance Report

Generated: 2025-08-17 14:44:35

## Summary

- Interfaces found: 8
- Implementations found: 19
- Violations found: 5

## Available Interfaces

### Any
**Module:** core.interfaces
**Methods:** 0


### Execution
**Module:** core.interfaces
**Methods:** 2

- `place_order(self, symbol: str, side: str, qty: float, **params: Any) -> Dict[str, Any]`
- `positions(self) -> List[Dict[str, Any]]`

### MarketData
**Module:** core.interfaces
**Methods:** 3

- `candles(self, symbol: str, timeframe: str, limit: int) -> pandas.core.frame.DataFrame`
- `ticker(self, symbol: str) -> Dict[str, float]`
- `volume_24h(self, symbol: str) -> float`

### NewsFeed
**Module:** core.interfaces
**Methods:** 2

- `macro_blockers(self, symbols: List[str]) -> Dict[str, bool]`
- `sentiment(self, symbols: List[str]) -> Dict[str, float]`

### OHLCV
**Module:** core.interfaces
**Methods:** 192

- `abs(self) -> 'Self'`
- `add(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `add_prefix(self, prefix: 'str', axis: 'Axis | None' = None) -> 'Self'`
- `add_suffix(self, suffix: 'str', axis: 'Axis | None' = None) -> 'Self'`
- `agg(self, func=None, axis: 'Axis' = 0, *args, **kwargs)`
- `aggregate(self, func=None, axis: 'Axis' = 0, *args, **kwargs)`
- `align(self, other: 'NDFrameT', join: 'AlignJoin' = 'outer', axis: 'Axis | None' = None, level: 'Level | None' = None, copy: 'bool_t | None' = None, fill_value: 'Hashable | None' = None, method: 'FillnaOptions | None | lib.NoDefault' = <no_default>, limit: 'int | None | lib.NoDefault' = <no_default>, fill_axis: 'Axis | lib.NoDefault' = <no_default>, broadcast_axis: 'Axis | None | lib.NoDefault' = <no_default>) -> 'tuple[Self, NDFrameT]'`
- `all(self, axis: 'Axis | None' = 0, bool_only: 'bool' = False, skipna: 'bool' = True, **kwargs) -> 'Series | bool'`
- `any(self, *, axis: 'Axis | None' = 0, bool_only: 'bool' = False, skipna: 'bool' = True, **kwargs) -> 'Series | bool'`
- `apply(self, func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type: "Literal['expand', 'reduce', 'broadcast'] | None" = None, args=(), by_row: "Literal[False, 'compat']" = 'compat', engine: "Literal['python', 'numba']" = 'python', engine_kwargs: 'dict[str, bool] | None' = None, **kwargs)`
- `applymap(self, func: 'PythonFuncType', na_action: 'NaAction | None' = None, **kwargs) -> 'DataFrame'`
- `asfreq(self, freq: 'Frequency', method: 'FillnaOptions | None' = None, how: "Literal['start', 'end'] | None" = None, normalize: 'bool_t' = False, fill_value: 'Hashable | None' = None) -> 'Self'`
- `asof(self, where, subset=None)`
- `assign(self, **kwargs) -> 'DataFrame'`
- `astype(self, dtype, copy: 'bool_t | None' = None, errors: 'IgnoreRaise' = 'raise') -> 'Self'`
- `at_time(self, time, asof: 'bool_t' = False, axis: 'Axis | None' = None) -> 'Self'`
- `backfill(self, *, axis: 'None | Axis' = None, inplace: 'bool_t' = False, limit: 'None | int' = None, downcast: 'dict | None | lib.NoDefault' = <no_default>) -> 'Self | None'`
- `between_time(self, start_time, end_time, inclusive: 'IntervalClosedType' = 'both', axis: 'Axis | None' = None) -> 'Self'`
- `bfill(self, *, axis: 'None | Axis' = None, inplace: 'bool_t' = False, limit: 'None | int' = None, limit_area: "Literal['inside', 'outside'] | None" = None, downcast: 'dict | None | lib.NoDefault' = <no_default>) -> 'Self | None'`
- `bool(self) -> 'bool_t'`
- `boxplot(self: 'DataFrame', column=None, by=None, ax=None, fontsize: 'int | None' = None, rot: 'int' = 0, grid: 'bool' = True, figsize: 'tuple[float, float] | None' = None, layout=None, return_type=None, backend=None, **kwargs)`
- `clip(self, lower=None, upper=None, *, axis: 'Axis | None' = None, inplace: 'bool_t' = False, **kwargs) -> 'Self | None'`
- `combine(self, other: 'DataFrame', func: 'Callable[[Series, Series], Series | Hashable]', fill_value=None, overwrite: 'bool' = True) -> 'DataFrame'`
- `combine_first(self, other: 'DataFrame') -> 'DataFrame'`
- `compare(self, other: 'DataFrame', align_axis: 'Axis' = 1, keep_shape: 'bool' = False, keep_equal: 'bool' = False, result_names: 'Suffixes' = ('self', 'other')) -> 'DataFrame'`
- `convert_dtypes(self, infer_objects: 'bool_t' = True, convert_string: 'bool_t' = True, convert_integer: 'bool_t' = True, convert_boolean: 'bool_t' = True, convert_floating: 'bool_t' = True, dtype_backend: 'DtypeBackend' = 'numpy_nullable') -> 'Self'`
- `copy(self, deep: 'bool_t | None' = True) -> 'Self'`
- `corr(self, method: 'CorrelationMethod' = 'pearson', min_periods: 'int' = 1, numeric_only: 'bool' = False) -> 'DataFrame'`
- `corrwith(self, other: 'DataFrame | Series', axis: 'Axis' = 0, drop: 'bool' = False, method: 'CorrelationMethod' = 'pearson', numeric_only: 'bool' = False) -> 'Series'`
- `count(self, axis: 'Axis' = 0, numeric_only: 'bool' = False)`
- `cov(self, min_periods: 'int | None' = None, ddof: 'int | None' = 1, numeric_only: 'bool' = False) -> 'DataFrame'`
- `cummax(self, axis: 'Axis | None' = None, skipna: 'bool' = True, *args, **kwargs)`
- `cummin(self, axis: 'Axis | None' = None, skipna: 'bool' = True, *args, **kwargs)`
- `cumprod(self, axis: 'Axis | None' = None, skipna: 'bool' = True, *args, **kwargs)`
- `cumsum(self, axis: 'Axis | None' = None, skipna: 'bool' = True, *args, **kwargs)`
- `describe(self, percentiles=None, include=None, exclude=None) -> 'Self'`
- `diff(self, periods: 'int' = 1, axis: 'Axis' = 0) -> 'DataFrame'`
- `div(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `divide(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `dot(self, other: 'AnyArrayLike | DataFrame') -> 'DataFrame | Series'`
- `drop(self, labels: 'IndexLabel | None' = None, *, axis: 'Axis' = 0, index: 'IndexLabel | None' = None, columns: 'IndexLabel | None' = None, level: 'Level | None' = None, inplace: 'bool' = False, errors: 'IgnoreRaise' = 'raise') -> 'DataFrame | None'`
- `drop_duplicates(self, subset: 'Hashable | Sequence[Hashable] | None' = None, *, keep: 'DropKeep' = 'first', inplace: 'bool' = False, ignore_index: 'bool' = False) -> 'DataFrame | None'`
- `droplevel(self, level: 'IndexLabel', axis: 'Axis' = 0) -> 'Self'`
- `dropna(self, *, axis: 'Axis' = 0, how: 'AnyAll | lib.NoDefault' = <no_default>, thresh: 'int | lib.NoDefault' = <no_default>, subset: 'IndexLabel | None' = None, inplace: 'bool' = False, ignore_index: 'bool' = False) -> 'DataFrame | None'`
- `duplicated(self, subset: 'Hashable | Sequence[Hashable] | None' = None, keep: 'DropKeep' = 'first') -> 'Series'`
- `eq(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `equals(self, other: 'object') -> 'bool_t'`
- `eval(self, expr: 'str', *, inplace: 'bool' = False, **kwargs) -> 'Any | None'`
- `ewm(self, com: 'float | None' = None, span: 'float | None' = None, halflife: 'float | TimedeltaConvertibleTypes | None' = None, alpha: 'float | None' = None, min_periods: 'int | None' = 0, adjust: 'bool_t' = True, ignore_na: 'bool_t' = False, axis: 'Axis | lib.NoDefault' = <no_default>, times: 'np.ndarray | DataFrame | Series | None' = None, method: "Literal['single', 'table']" = 'single') -> 'ExponentialMovingWindow'`
- `expanding(self, min_periods: 'int' = 1, axis: 'Axis | lib.NoDefault' = <no_default>, method: "Literal['single', 'table']" = 'single') -> 'Expanding'`
- `explode(self, column: 'IndexLabel', ignore_index: 'bool' = False) -> 'DataFrame'`
- `ffill(self, *, axis: 'None | Axis' = None, inplace: 'bool_t' = False, limit: 'None | int' = None, limit_area: "Literal['inside', 'outside'] | None" = None, downcast: 'dict | None | lib.NoDefault' = <no_default>) -> 'Self | None'`
- `fillna(self, value: 'Hashable | Mapping | Series | DataFrame | None' = None, *, method: 'FillnaOptions | None' = None, axis: 'Axis | None' = None, inplace: 'bool_t' = False, limit: 'int | None' = None, downcast: 'dict | None | lib.NoDefault' = <no_default>) -> 'Self | None'`
- `filter(self, items=None, like: 'str | None' = None, regex: 'str | None' = None, axis: 'Axis | None' = None) -> 'Self'`
- `first(self, offset) -> 'Self'`
- `first_valid_index(self) -> 'Hashable | None'`
- `floordiv(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `from_dict(data: 'dict', orient: 'FromDictOrient' = 'columns', dtype: 'Dtype | None' = None, columns: 'Axes | None' = None) -> 'DataFrame'`
- `from_records(data, index=None, exclude=None, columns=None, coerce_float: 'bool' = False, nrows: 'int | None' = None) -> 'DataFrame'`
- `ge(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `get(self, key, default=None)`
- `groupby(self, by=None, axis: 'Axis | lib.NoDefault' = <no_default>, level: 'IndexLabel | None' = None, as_index: 'bool' = True, sort: 'bool' = True, group_keys: 'bool' = True, observed: 'bool | lib.NoDefault' = <no_default>, dropna: 'bool' = True) -> 'DataFrameGroupBy'`
- `gt(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `head(self, n: 'int' = 5) -> 'Self'`
- `hist(data: 'DataFrame', column: 'IndexLabel | None' = None, by=None, grid: 'bool' = True, xlabelsize: 'int | None' = None, xrot: 'float | None' = None, ylabelsize: 'int | None' = None, yrot: 'float | None' = None, ax=None, sharex: 'bool' = False, sharey: 'bool' = False, figsize: 'tuple[int, int] | None' = None, layout: 'tuple[int, int] | None' = None, bins: 'int | Sequence[int]' = 10, backend: 'str | None' = None, legend: 'bool' = False, **kwargs)`
- `idxmax(self, axis: 'Axis' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False) -> 'Series'`
- `idxmin(self, axis: 'Axis' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False) -> 'Series'`
- `infer_objects(self, copy: 'bool_t | None' = None) -> 'Self'`
- `info(self, verbose: 'bool | None' = None, buf: 'WriteBuffer[str] | None' = None, max_cols: 'int | None' = None, memory_usage: 'bool | str | None' = None, show_counts: 'bool | None' = None) -> 'None'`
- `insert(self, loc: 'int', column: 'Hashable', value: 'Scalar | AnyArrayLike', allow_duplicates: 'bool | lib.NoDefault' = <no_default>) -> 'None'`
- `interpolate(self, method: 'InterpolateOptions' = 'linear', *, axis: 'Axis' = 0, limit: 'int | None' = None, inplace: 'bool_t' = False, limit_direction: "Literal['forward', 'backward', 'both'] | None" = None, limit_area: "Literal['inside', 'outside'] | None" = None, downcast: "Literal['infer'] | None | lib.NoDefault" = <no_default>, **kwargs) -> 'Self | None'`
- `isetitem(self, loc, value) -> 'None'`
- `isin(self, values: 'Series | DataFrame | Sequence | Mapping') -> 'DataFrame'`
- `isna(self) -> 'DataFrame'`
- `isnull(self) -> 'DataFrame'`
- `items(self) -> 'Iterable[tuple[Hashable, Series]]'`
- `iterrows(self) -> 'Iterable[tuple[Hashable, Series]]'`
- `itertuples(self, index: 'bool' = True, name: 'str | None' = 'Pandas') -> 'Iterable[tuple[Any, ...]]'`
- `join(self, other: 'DataFrame | Series | Iterable[DataFrame | Series]', on: 'IndexLabel | None' = None, how: 'MergeHow' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False, validate: 'JoinValidate | None' = None) -> 'DataFrame'`
- `keys(self) -> 'Index'`
- `kurt(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `kurtosis(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `last(self, offset) -> 'Self'`
- `last_valid_index(self) -> 'Hashable | None'`
- `le(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `lt(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `map(self, func: 'PythonFuncType', na_action: 'str | None' = None, **kwargs) -> 'DataFrame'`
- `mask(self, cond, other=<no_default>, *, inplace: 'bool_t' = False, axis: 'Axis | None' = None, level: 'Level | None' = None) -> 'Self | None'`
- `max(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `mean(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `median(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `melt(self, id_vars=None, value_vars=None, var_name=None, value_name: 'Hashable' = 'value', col_level: 'Level | None' = None, ignore_index: 'bool' = True) -> 'DataFrame'`
- `memory_usage(self, index: 'bool' = True, deep: 'bool' = False) -> 'Series'`
- `merge(self, right: 'DataFrame | Series', how: 'MergeHow' = 'inner', on: 'IndexLabel | AnyArrayLike | None' = None, left_on: 'IndexLabel | AnyArrayLike | None' = None, right_on: 'IndexLabel | AnyArrayLike | None' = None, left_index: 'bool' = False, right_index: 'bool' = False, sort: 'bool' = False, suffixes: 'Suffixes' = ('_x', '_y'), copy: 'bool | None' = None, indicator: 'str | bool' = False, validate: 'MergeValidate | None' = None) -> 'DataFrame'`
- `min(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `mod(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `mode(self, axis: 'Axis' = 0, numeric_only: 'bool' = False, dropna: 'bool' = True) -> 'DataFrame'`
- `mul(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `multiply(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `ne(self, other, axis: 'Axis' = 'columns', level=None) -> 'DataFrame'`
- `nlargest(self, n: 'int', columns: 'IndexLabel', keep: 'NsmallestNlargestKeep' = 'first') -> 'DataFrame'`
- `notna(self) -> 'DataFrame'`
- `notnull(self) -> 'DataFrame'`
- `nsmallest(self, n: 'int', columns: 'IndexLabel', keep: 'NsmallestNlargestKeep' = 'first') -> 'DataFrame'`
- `nunique(self, axis: 'Axis' = 0, dropna: 'bool' = True) -> 'Series'`
- `pad(self, *, axis: 'None | Axis' = None, inplace: 'bool_t' = False, limit: 'None | int' = None, downcast: 'dict | None | lib.NoDefault' = <no_default>) -> 'Self | None'`
- `pct_change(self, periods: 'int' = 1, fill_method: 'FillnaOptions | None | lib.NoDefault' = <no_default>, limit: 'int | None | lib.NoDefault' = <no_default>, freq=None, **kwargs) -> 'Self'`
- `pipe(self, func: 'Callable[..., T] | tuple[Callable[..., T], str]', *args, **kwargs) -> 'T'`
- `pivot(self, *, columns, index=<no_default>, values=<no_default>) -> 'DataFrame'`
- `pivot_table(self, values=None, index=None, columns=None, aggfunc: 'AggFuncType' = 'mean', fill_value=None, margins: 'bool' = False, dropna: 'bool' = True, margins_name: 'Level' = 'All', observed: 'bool | lib.NoDefault' = <no_default>, sort: 'bool' = True) -> 'DataFrame'`
- `plot(data: 'Series | DataFrame') -> 'None'`
- `pop(self, item: 'Hashable') -> 'Series'`
- `pow(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `prod(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, min_count: 'int' = 0, **kwargs)`
- `product(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, min_count: 'int' = 0, **kwargs)`
- `quantile(self, q: 'float | AnyArrayLike | Sequence[float]' = 0.5, axis: 'Axis' = 0, numeric_only: 'bool' = False, interpolation: 'QuantileInterpolation' = 'linear', method: "Literal['single', 'table']" = 'single') -> 'Series | DataFrame'`
- `query(self, expr: 'str', *, inplace: 'bool' = False, **kwargs) -> 'DataFrame | None'`
- `radd(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rank(self, axis: 'Axis' = 0, method: "Literal['average', 'min', 'max', 'first', 'dense']" = 'average', numeric_only: 'bool_t' = False, na_option: "Literal['keep', 'top', 'bottom']" = 'keep', ascending: 'bool_t' = True, pct: 'bool_t' = False) -> 'Self'`
- `rdiv(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `reindex(self, labels=None, *, index=None, columns=None, axis: 'Axis | None' = None, method: 'ReindexMethod | None' = None, copy: 'bool | None' = None, level: 'Level | None' = None, fill_value: 'Scalar | None' = nan, limit: 'int | None' = None, tolerance=None) -> 'DataFrame'`
- `reindex_like(self, other, method: "Literal['backfill', 'bfill', 'pad', 'ffill', 'nearest'] | None" = None, copy: 'bool_t | None' = None, limit: 'int | None' = None, tolerance=None) -> 'Self'`
- `rename(self, mapper: 'Renamer | None' = None, *, index: 'Renamer | None' = None, columns: 'Renamer | None' = None, axis: 'Axis | None' = None, copy: 'bool | None' = None, inplace: 'bool' = False, level: 'Level | None' = None, errors: 'IgnoreRaise' = 'ignore') -> 'DataFrame | None'`
- `rename_axis(self, mapper: 'IndexLabel | lib.NoDefault' = <no_default>, *, index=<no_default>, columns=<no_default>, axis: 'Axis' = 0, copy: 'bool_t | None' = None, inplace: 'bool_t' = False) -> 'Self | None'`
- `reorder_levels(self, order: 'Sequence[int | str]', axis: 'Axis' = 0) -> 'DataFrame'`
- `replace(self, to_replace=None, value=<no_default>, *, inplace: 'bool_t' = False, limit: 'int | None' = None, regex: 'bool_t' = False, method: "Literal['pad', 'ffill', 'bfill'] | lib.NoDefault" = <no_default>) -> 'Self | None'`
- `resample(self, rule, axis: 'Axis | lib.NoDefault' = <no_default>, closed: "Literal['right', 'left'] | None" = None, label: "Literal['right', 'left'] | None" = None, convention: "Literal['start', 'end', 's', 'e'] | lib.NoDefault" = <no_default>, kind: "Literal['timestamp', 'period'] | None | lib.NoDefault" = <no_default>, on: 'Level | None' = None, level: 'Level | None' = None, origin: 'str | TimestampConvertibleTypes' = 'start_day', offset: 'TimedeltaConvertibleTypes | None' = None, group_keys: 'bool_t' = False) -> 'Resampler'`
- `reset_index(self, level: 'IndexLabel | None' = None, *, drop: 'bool' = False, inplace: 'bool' = False, col_level: 'Hashable' = 0, col_fill: 'Hashable' = '', allow_duplicates: 'bool | lib.NoDefault' = <no_default>, names: 'Hashable | Sequence[Hashable] | None' = None) -> 'DataFrame | None'`
- `rfloordiv(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rmod(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rmul(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rolling(self, window: 'int | dt.timedelta | str | BaseOffset | BaseIndexer', min_periods: 'int | None' = None, center: 'bool_t' = False, win_type: 'str | None' = None, on: 'str | None' = None, axis: 'Axis | lib.NoDefault' = <no_default>, closed: 'IntervalClosedType | None' = None, step: 'int | None' = None, method: 'str' = 'single') -> 'Window | Rolling'`
- `round(self, decimals: 'int | dict[IndexLabel, int] | Series' = 0, *args, **kwargs) -> 'DataFrame'`
- `rpow(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rsub(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `rtruediv(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `sample(self, n: 'int | None' = None, frac: 'float | None' = None, replace: 'bool_t' = False, weights=None, random_state: 'RandomState | None' = None, axis: 'Axis | None' = None, ignore_index: 'bool_t' = False) -> 'Self'`
- `select_dtypes(self, include=None, exclude=None) -> 'Self'`
- `sem(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, ddof: 'int' = 1, numeric_only: 'bool' = False, **kwargs)`
- `set_axis(self, labels, *, axis: 'Axis' = 0, copy: 'bool | None' = None) -> 'DataFrame'`
- `set_flags(self, *, copy: 'bool_t' = False, allows_duplicate_labels: 'bool_t | None' = None) -> 'Self'`
- `set_index(self, keys, *, drop: 'bool' = True, append: 'bool' = False, inplace: 'bool' = False, verify_integrity: 'bool' = False) -> 'DataFrame | None'`
- `shift(self, periods: 'int | Sequence[int]' = 1, freq: 'Frequency | None' = None, axis: 'Axis' = 0, fill_value: 'Hashable' = <no_default>, suffix: 'str | None' = None) -> 'DataFrame'`
- `skew(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, **kwargs)`
- `sort_index(self, *, axis: 'Axis' = 0, level: 'IndexLabel | None' = None, ascending: 'bool | Sequence[bool]' = True, inplace: 'bool' = False, kind: 'SortKind' = 'quicksort', na_position: 'NaPosition' = 'last', sort_remaining: 'bool' = True, ignore_index: 'bool' = False, key: 'IndexKeyFunc | None' = None) -> 'DataFrame | None'`
- `sort_values(self, by: 'IndexLabel', *, axis: 'Axis' = 0, ascending: 'bool | list[bool] | tuple[bool, ...]' = True, inplace: 'bool' = False, kind: 'SortKind' = 'quicksort', na_position: 'str' = 'last', ignore_index: 'bool' = False, key: 'ValueKeyFunc | None' = None) -> 'DataFrame | None'`
- `sparse(data=None) -> 'None'`
- `squeeze(self, axis: 'Axis | None' = None)`
- `stack(self, level: 'IndexLabel' = -1, dropna: 'bool | lib.NoDefault' = <no_default>, sort: 'bool | lib.NoDefault' = <no_default>, future_stack: 'bool' = False)`
- `std(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, ddof: 'int' = 1, numeric_only: 'bool' = False, **kwargs)`
- `sub(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `subtract(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `sum(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, numeric_only: 'bool' = False, min_count: 'int' = 0, **kwargs)`
- `swapaxes(self, axis1: 'Axis', axis2: 'Axis', copy: 'bool_t | None' = None) -> 'Self'`
- `swaplevel(self, i: 'Axis' = -2, j: 'Axis' = -1, axis: 'Axis' = 0) -> 'DataFrame'`
- `tail(self, n: 'int' = 5) -> 'Self'`
- `take(self, indices, axis: 'Axis' = 0, **kwargs) -> 'Self'`
- `to_clipboard(self, *, excel: 'bool_t' = True, sep: 'str | None' = None, **kwargs) -> 'None'`
- `to_csv(self, path_or_buf: 'FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None' = None, *, sep: 'str' = ',', na_rep: 'str' = '', float_format: 'str | Callable | None' = None, columns: 'Sequence[Hashable] | None' = None, header: 'bool_t | list[str]' = True, index: 'bool_t' = True, index_label: 'IndexLabel | None' = None, mode: 'str' = 'w', encoding: 'str | None' = None, compression: 'CompressionOptions' = 'infer', quoting: 'int | None' = None, quotechar: 'str' = '"', lineterminator: 'str | None' = None, chunksize: 'int | None' = None, date_format: 'str | None' = None, doublequote: 'bool_t' = True, escapechar: 'str | None' = None, decimal: 'str' = '.', errors: 'OpenFileErrors' = 'strict', storage_options: 'StorageOptions | None' = None) -> 'str | None'`
- `to_dict(self, orient: "Literal['dict', 'list', 'series', 'split', 'tight', 'records', 'index']" = 'dict', *, into: 'type[MutableMappingT] | MutableMappingT' = <class 'dict'>, index: 'bool' = True) -> 'MutableMappingT | list[MutableMappingT]'`
- `to_excel(self, excel_writer: 'FilePath | WriteExcelBuffer | ExcelWriter', *, sheet_name: 'str' = 'Sheet1', na_rep: 'str' = '', float_format: 'str | None' = None, columns: 'Sequence[Hashable] | None' = None, header: 'Sequence[Hashable] | bool_t' = True, index: 'bool_t' = True, index_label: 'IndexLabel | None' = None, startrow: 'int' = 0, startcol: 'int' = 0, engine: "Literal['openpyxl', 'xlsxwriter'] | None" = None, merge_cells: 'bool_t' = True, inf_rep: 'str' = 'inf', freeze_panes: 'tuple[int, int] | None' = None, storage_options: 'StorageOptions | None' = None, engine_kwargs: 'dict[str, Any] | None' = None) -> 'None'`
- `to_feather(self, path: 'FilePath | WriteBuffer[bytes]', **kwargs) -> 'None'`
- `to_gbq(self, destination_table: 'str', *, project_id: 'str | None' = None, chunksize: 'int | None' = None, reauth: 'bool' = False, if_exists: 'ToGbqIfexist' = 'fail', auth_local_webserver: 'bool' = True, table_schema: 'list[dict[str, str]] | None' = None, location: 'str | None' = None, progress_bar: 'bool' = True, credentials=None) -> 'None'`
- `to_hdf(self, path_or_buf: 'FilePath | HDFStore', *, key: 'str', mode: "Literal['a', 'w', 'r+']" = 'a', complevel: 'int | None' = None, complib: "Literal['zlib', 'lzo', 'bzip2', 'blosc'] | None" = None, append: 'bool_t' = False, format: "Literal['fixed', 'table'] | None" = None, index: 'bool_t' = True, min_itemsize: 'int | dict[str, int] | None' = None, nan_rep=None, dropna: 'bool_t | None' = None, data_columns: 'Literal[True] | list[str] | None' = None, errors: 'OpenFileErrors' = 'strict', encoding: 'str' = 'UTF-8') -> 'None'`
- `to_html(self, buf: 'FilePath | WriteBuffer[str] | None' = None, *, columns: 'Axes | None' = None, col_space: 'ColspaceArgType | None' = None, header: 'bool' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'FormattersType | None' = None, float_format: 'FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool | str' = False, decimal: 'str' = '.', bold_rows: 'bool' = True, classes: 'str | list | tuple | None' = None, escape: 'bool' = True, notebook: 'bool' = False, border: 'int | bool | None' = None, table_id: 'str | None' = None, render_links: 'bool' = False, encoding: 'str | None' = None) -> 'str | None'`
- `to_json(self, path_or_buf: 'FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None' = None, *, orient: "Literal['split', 'records', 'index', 'table', 'columns', 'values'] | None" = None, date_format: 'str | None' = None, double_precision: 'int' = 10, force_ascii: 'bool_t' = True, date_unit: 'TimeUnit' = 'ms', default_handler: 'Callable[[Any], JSONSerializable] | None' = None, lines: 'bool_t' = False, compression: 'CompressionOptions' = 'infer', index: 'bool_t | None' = None, indent: 'int | None' = None, storage_options: 'StorageOptions | None' = None, mode: "Literal['a', 'w']" = 'w') -> 'str | None'`
- `to_latex(self, buf: 'FilePath | WriteBuffer[str] | None' = None, *, columns: 'Sequence[Hashable] | None' = None, header: 'bool_t | SequenceNotStr[str]' = True, index: 'bool_t' = True, na_rep: 'str' = 'NaN', formatters: 'FormattersType | None' = None, float_format: 'FloatFormatType | None' = None, sparsify: 'bool_t | None' = None, index_names: 'bool_t' = True, bold_rows: 'bool_t' = False, column_format: 'str | None' = None, longtable: 'bool_t | None' = None, escape: 'bool_t | None' = None, encoding: 'str | None' = None, decimal: 'str' = '.', multicolumn: 'bool_t | None' = None, multicolumn_format: 'str | None' = None, multirow: 'bool_t | None' = None, caption: 'str | tuple[str, str] | None' = None, label: 'str | None' = None, position: 'str | None' = None) -> 'str | None'`
- `to_markdown(self, buf: 'FilePath | WriteBuffer[str] | None' = None, *, mode: 'str' = 'wt', index: 'bool' = True, storage_options: 'StorageOptions | None' = None, **kwargs) -> 'str | None'`
- `to_numpy(self, dtype: 'npt.DTypeLike | None' = None, copy: 'bool' = False, na_value: 'object' = <no_default>) -> 'np.ndarray'`
- `to_orc(self, path: 'FilePath | WriteBuffer[bytes] | None' = None, *, engine: "Literal['pyarrow']" = 'pyarrow', index: 'bool | None' = None, engine_kwargs: 'dict[str, Any] | None' = None) -> 'bytes | None'`
- `to_parquet(self, path: 'FilePath | WriteBuffer[bytes] | None' = None, *, engine: "Literal['auto', 'pyarrow', 'fastparquet']" = 'auto', compression: 'str | None' = 'snappy', index: 'bool | None' = None, partition_cols: 'list[str] | None' = None, storage_options: 'StorageOptions | None' = None, **kwargs) -> 'bytes | None'`
- `to_period(self, freq: 'Frequency | None' = None, axis: 'Axis' = 0, copy: 'bool | None' = None) -> 'DataFrame'`
- `to_pickle(self, path: 'FilePath | WriteBuffer[bytes]', *, compression: 'CompressionOptions' = 'infer', protocol: 'int' = 5, storage_options: 'StorageOptions | None' = None) -> 'None'`
- `to_records(self, index: 'bool' = True, column_dtypes=None, index_dtypes=None) -> 'np.rec.recarray'`
- `to_sql(self, name: 'str', con, *, schema: 'str | None' = None, if_exists: "Literal['fail', 'replace', 'append']" = 'fail', index: 'bool_t' = True, index_label: 'IndexLabel | None' = None, chunksize: 'int | None' = None, dtype: 'DtypeArg | None' = None, method: "Literal['multi'] | Callable | None" = None) -> 'int | None'`
- `to_stata(self, path: 'FilePath | WriteBuffer[bytes]', *, convert_dates: 'dict[Hashable, str] | None' = None, write_index: 'bool' = True, byteorder: 'ToStataByteorder | None' = None, time_stamp: 'datetime.datetime | None' = None, data_label: 'str | None' = None, variable_labels: 'dict[Hashable, str] | None' = None, version: 'int | None' = 114, convert_strl: 'Sequence[Hashable] | None' = None, compression: 'CompressionOptions' = 'infer', storage_options: 'StorageOptions | None' = None, value_labels: 'dict[Hashable, dict[float, str]] | None' = None) -> 'None'`
- `to_string(self, buf: 'FilePath | WriteBuffer[str] | None' = None, *, columns: 'Axes | None' = None, col_space: 'int | list[int] | dict[Hashable, int] | None' = None, header: 'bool | SequenceNotStr[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, min_rows: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None) -> 'str | None'`
- `to_timestamp(self, freq: 'Frequency | None' = None, how: 'ToTimestampHow' = 'start', axis: 'Axis' = 0, copy: 'bool | None' = None) -> 'DataFrame'`
- `to_xarray(self)`
- `to_xml(self, path_or_buffer: 'FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None' = None, *, index: 'bool' = True, root_name: 'str | None' = 'data', row_name: 'str | None' = 'row', na_rep: 'str | None' = None, attr_cols: 'list[str] | None' = None, elem_cols: 'list[str] | None' = None, namespaces: 'dict[str | None, str] | None' = None, prefix: 'str | None' = None, encoding: 'str' = 'utf-8', xml_declaration: 'bool | None' = True, pretty_print: 'bool | None' = True, parser: 'XMLParsers | None' = 'lxml', stylesheet: 'FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None' = None, compression: 'CompressionOptions' = 'infer', storage_options: 'StorageOptions | None' = None) -> 'str | None'`
- `transform(self, func: 'AggFuncType', axis: 'Axis' = 0, *args, **kwargs) -> 'DataFrame'`
- `transpose(self, *args, copy: 'bool' = False) -> 'DataFrame'`
- `truediv(self, other, axis: 'Axis' = 'columns', level=None, fill_value=None) -> 'DataFrame'`
- `truncate(self, before=None, after=None, axis: 'Axis | None' = None, copy: 'bool_t | None' = None) -> 'Self'`
- `tz_convert(self, tz, axis: 'Axis' = 0, level=None, copy: 'bool_t | None' = None) -> 'Self'`
- `tz_localize(self, tz, axis: 'Axis' = 0, level=None, copy: 'bool_t | None' = None, ambiguous: 'TimeAmbiguous' = 'raise', nonexistent: 'TimeNonexistent' = 'raise') -> 'Self'`
- `unstack(self, level: 'IndexLabel' = -1, fill_value=None, sort: 'bool' = True)`
- `update(self, other, join: 'UpdateJoin' = 'left', overwrite: 'bool' = True, filter_func=None, errors: 'IgnoreRaise' = 'ignore') -> 'None'`
- `value_counts(self, subset: 'IndexLabel | None' = None, normalize: 'bool' = False, sort: 'bool' = True, ascending: 'bool' = False, dropna: 'bool' = True) -> 'Series'`
- `var(self, axis: 'Axis | None' = 0, skipna: 'bool' = True, ddof: 'int' = 1, numeric_only: 'bool' = False, **kwargs)`
- `where(self, cond, other=nan, *, inplace: 'bool_t' = False, axis: 'Axis | None' = None, level: 'Level | None' = None) -> 'Self | None'`
- `xs(self, key: 'IndexLabel', axis: 'Axis' = 0, level: 'IndexLabel | None' = None, drop_level: 'bool_t' = True) -> 'Self'`

### Protocol
**Module:** core.interfaces
**Methods:** 0


### ValidationRunner
**Module:** core.interfaces
**Methods:** 1

- `approved(self, strategy_id: str, market: str) -> Tuple[bool, Dict[str, Any]]`

### WalletSync
**Module:** core.interfaces
**Methods:** 1

- `subledger_equity(self) -> Dict[str, float]`

## Implementation Status

### IbkrMarketData
**File:** adapters/ibkr_exec.py
**Implements:** MarketData
**Status:** ✅ Conformant

### MarketData
**File:** adapters/null_adapters.py
**Implements:** MarketData
**Status:** ✅ Conformant

### OHLCV
**File:** adapters/ibkr_market.py
**Implements:** OHLCV
**Status:** ❌ 2 violations

- ❌ Method 'plot' required by OHLCV is missing in OHLCV
- ❌ Method 'sparse' required by OHLCV is missing in OHLCV

### Any
**File:** adapters/ibkr_exec.py
**Implements:** Any
**Status:** ✅ Conformant

### Execution
**File:** adapters/ibkr_exec.py
**Implements:** Execution
**Status:** ✅ Conformant

### NewsFeed
**File:** adapters/news_rss.py
**Implements:** NewsFeed
**Status:** ✅ Conformant

### NullExecution
**File:** adapters/null_adapters.py
**Implements:** Execution
**Status:** ✅ Conformant

### NullMarketData
**File:** adapters/null_adapters.py
**Implements:** MarketData
**Status:** ✅ Conformant

### NullNewsFeed
**File:** adapters/null_adapters.py
**Implements:** NewsFeed
**Status:** ✅ Conformant

### NullValidationRunner
**File:** adapters/null_adapters.py
**Implements:** ValidationRunner
**Status:** ✅ Conformant

### NullWalletSync
**File:** adapters/null_adapters.py
**Implements:** WalletSync
**Status:** ✅ Conformant

### ValidationRunner
**File:** adapters/null_adapters.py
**Implements:** ValidationRunner
**Status:** ✅ Conformant

### WalletSync
**File:** adapters/composite_wallet.py
**Implements:** WalletSync
**Status:** ✅ Conformant

### NewsRssAdapter
**File:** adapters/news_rss.py
**Implements:** NewsFeed
**Status:** ✅ Conformant

### BybitWalletSync
**File:** adapters/wallet_bybit.py
**Implements:** WalletSync
**Status:** ✅ Conformant

### IbkrWalletSync
**File:** adapters/wallet_ibkr.py
**Implements:** WalletSync
**Status:** ✅ Conformant

### CompositeWalletSync
**File:** adapters/composite_wallet.py
**Implements:** WalletSync
**Status:** ✅ Conformant

### IbkrExecution
**File:** adapters/ibkr_exec.py
**Implements:** Execution
**Status:** ✅ Conformant

### MarketOrder
**File:** adapters/ibkr_exec.py
**Implements:** MarketData
**Status:** ❌ 3 violations

- ❌ Method 'candles' required by MarketData is missing in MarketOrder
- ❌ Method 'ticker' required by MarketData is missing in MarketOrder
- ❌ Method 'volume_24h' required by MarketData is missing in MarketOrder

## Detailed Violations

### Missing Method

**OHLCV**
- File: adapters/ibkr_market.py
- Interface: OHLCV
- Method: plot
- Issue: Method 'plot' required by OHLCV is missing in OHLCV

**OHLCV**
- File: adapters/ibkr_market.py
- Interface: OHLCV
- Method: sparse
- Issue: Method 'sparse' required by OHLCV is missing in OHLCV

**MarketOrder**
- File: adapters/ibkr_exec.py
- Interface: MarketData
- Method: candles
- Issue: Method 'candles' required by MarketData is missing in MarketOrder

**MarketOrder**
- File: adapters/ibkr_exec.py
- Interface: MarketData
- Method: ticker
- Issue: Method 'ticker' required by MarketData is missing in MarketOrder

**MarketOrder**
- File: adapters/ibkr_exec.py
- Interface: MarketData
- Method: volume_24h
- Issue: Method 'volume_24h' required by MarketData is missing in MarketOrder

