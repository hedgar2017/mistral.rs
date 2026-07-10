use indicatif::{
    MultiProgress, ProgressBar, ProgressBarIter, ProgressDrawTarget, ProgressIterator,
    ProgressStyle,
};
use mistralrs_quant::get_immediate_isq;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::iter::Iterator;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tqdm::Iter;

static PROGRESS_SUPPRESS_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Observer invoked on each [`NiceProgressBar`] step during model load, with
/// `(label, position, total)`. A host (e.g. a GUI) registers one via
/// [`set_load_progress_observer`] to mirror the load bar without the terminal
/// widget; `silent` suppression of the indicatif draw is independent of this.
type LoadProgressObserver = Arc<dyn Fn(&str, u64, u64) + Send + Sync>;

static LOAD_PROGRESS_OBSERVER: RwLock<Option<LoadProgressObserver>> = RwLock::new(None);

/// Register the load-progress observer, replacing any previous one.
pub fn set_load_progress_observer<F>(observer: F)
where
    F: Fn(&str, u64, u64) + Send + Sync + 'static,
{
    if let Ok(mut slot) = LOAD_PROGRESS_OBSERVER.write() {
        *slot = Some(Arc::new(observer));
    }
}

/// Clear the load-progress observer.
pub fn clear_load_progress_observer() {
    if let Ok(mut slot) = LOAD_PROGRESS_OBSERVER.write() {
        *slot = None;
    }
}

fn notify_load_progress(label: &str, position: u64, total: u64) {
    if let Ok(slot) = LOAD_PROGRESS_OBSERVER.read() {
        if let Some(observer) = slot.as_ref() {
            observer(label, position, total);
        }
    }
}

/// RAII guard that suppresses progress bar drawing while it is alive.
pub struct ProgressScopeGuard {
    suppressed: bool,
}

impl ProgressScopeGuard {
    pub fn new(silent: bool) -> Self {
        if silent {
            PROGRESS_SUPPRESS_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        Self { suppressed: silent }
    }
}

impl Drop for ProgressScopeGuard {
    fn drop(&mut self) {
        if self.suppressed {
            PROGRESS_SUPPRESS_COUNT.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

#[inline]
pub fn progress_suppressed() -> bool {
    PROGRESS_SUPPRESS_COUNT.load(Ordering::SeqCst) > 0
}

#[inline]
pub fn configure_progress_bar(bar: &ProgressBar) {
    if progress_suppressed() {
        bar.set_draw_target(ProgressDrawTarget::hidden());
    }
}

pub fn new_multi_progress() -> MultiProgress {
    let multi = MultiProgress::new();
    if progress_suppressed() {
        multi.set_draw_target(ProgressDrawTarget::hidden());
    }
    multi
}

// Optionally display a progress bar via the `tqdm` crate:
// Usage: `iter.with_progress(true)`
// Similar to the `iter.tqdm()` feature except this supports opt-in via parameter.
pub trait IterWithProgress<'a, T>: Iterator<Item = T> + 'a {
    fn with_progress(self, is_silent: bool) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: Sized,
    {
        // TODO: Should `is_silent` instead be referenced as a global read-only state? (`AtomicBool`)
        if is_silent {
            Box::new(self)
        } else {
            Box::new(self.tqdm())
        }
    }
}

impl<'a, T: Iterator + 'a> IterWithProgress<'a, T::Item> for T {}

/// Nice progress bar with over an iterator and a message.
/// COLOR is one of r,g,b
pub struct NiceProgressBar<'a, T: ExactSizeIterator, const COLOR: char = 'b'>(
    pub T,
    pub &'static str,
    pub &'a MultiProgress,
);

/// Drives the indicatif bar AND notifies the global load-progress observer on
/// each step (label + 1-based position + total), so a host can mirror the load
/// even when the terminal bar is suppressed.
pub struct ObservedProgress<T: ExactSizeIterator> {
    inner: ProgressBarIter<T>,
    label: &'static str,
    position: u64,
    total: u64,
}

impl<T: ExactSizeIterator> Iterator for ObservedProgress<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();
        if item.is_some() {
            self.position += 1;
            notify_load_progress(self.label, self.position, self.total);
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: ExactSizeIterator> ExactSizeIterator for ObservedProgress<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: ExactSizeIterator, const COLOR: char> IntoIterator for NiceProgressBar<'_, T, COLOR> {
    type IntoIter = ObservedProgress<T>;
    type Item = T::Item;

    fn into_iter(self) -> Self::IntoIter {
        let color = match COLOR {
            'b' => "blue",
            'g' => "green",
            'r' => "red",
            other => panic!("Color char `{other}` not supported"),
        };
        let label = self.1;
        let total = self.0.len() as u64;
        let bar = ProgressBar::new(total);
        configure_progress_bar(&bar);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{}: [{{elapsed_precise}}] [{{bar:40.{color}/{color}}}] {{pos}}/{{len}} ({{eta}})",
                    self.1
                ))
                .unwrap()
                .progress_chars("#>-"),
        );

        // Add to the multi progress
        self.2.add(bar.clone());

        ObservedProgress {
            inner: self.0.progress_with(bar),
            label,
            position: 0,
            total,
        }
    }
}

/// Parallel iterator with progress reporting.
pub struct ParProgress<I> {
    iter: I,
    bar: ProgressBar,
}

impl<I> ParallelIterator for ParProgress<I>
where
    I: ParallelIterator,
    I::Item: Send,
{
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.drive_unindexed(consumer)
    }
}

impl<I> IndexedParallelIterator for ParProgress<I>
where
    I: IndexedParallelIterator,
    I::Item: Send,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.drive(consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.with_producer(callback)
    }
}

impl<'a, T, const COLOR: char> IntoParallelIterator for NiceProgressBar<'a, T, COLOR>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
{
    type Iter = ParProgress<T::Iter>;
    type Item = <T as IntoParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        let color = match COLOR {
            'b' => "blue",
            'g' => "green",
            'r' => "red",
            other => panic!("Color char `{other}` not supported"),
        };
        let bar = ProgressBar::new(self.0.len() as u64);
        configure_progress_bar(&bar);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{}: [{{elapsed_precise}}] [{{bar:40.{color}/{color}}}] {{pos}}/{{len}} ({{eta}})",
                    self.1
                ))
                .unwrap()
                .progress_chars("#>-"),
        );
        self.2.add(bar.clone());
        ParProgress {
            iter: self.0.into_par_iter(),
            bar,
        }
    }
}

impl<'a, T, const COLOR: char> NiceProgressBar<'a, T, COLOR>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
    T: IntoParallelIterator<Item = <T as Iterator>::Item>,
{
    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `is_parallel`: If true, uses Rayon parallel iteration; otherwise uses sequential iteration.
    /// - `f`: A closure to apply to each item.
    pub fn run<F, U>(self, _is_parallel: bool, f: F) -> candle_core::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> candle_core::Result<U> + Sync + Send,
        U: Send,
    {
        // if is_parallel {
        //     self.into_par_iter().map(f).collect()
        // } else {
        //     self.into_iter().map(f).collect()
        // }
        self.into_iter().map(f).collect()
    }

    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `f`: A closure to apply to each item.
    pub fn par_iter_if_isq<F, U>(self, f: F) -> candle_core::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> candle_core::Result<U> + Sync + Send,
        U: Send,
    {
        self.run(get_immediate_isq().is_some_and(|x| x.ty.is_some()), f)
    }
}
