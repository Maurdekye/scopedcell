//! A variant of [`std::cell::Cell`] which is more performant for large heap-allocated data.
//!
//! `Cell` is typically best used with small types that implement [`Copy`]. The primary way `Cell` maintains
//! memory safety is by never providing a direct reference to the contained value within itself; only copies
//! of that data. For this reason, `Cell` is generally only used with small types that are cheap to copy,
//! like primitives and small enums.
//!
//! If one wants to place a larger, heap-allocated type within a value that has internal mutability, they might
//! opt to use a [`std::cell::RefCell`] instead, which provides compile-time borrow-checking, allowing one to
//! acquire shared or mutable references to a contained inner value, even if the containing type is immutable.
//! However, keeping track of those borrows and ensuring that your program's runtime does not invalidate rust's
//! borrowing rules during execution can be a tiring and often unnecessary hassle.
//! 
//! `ScopedCell` provides a middleground between these two options. It's somewhat more restrictive than a `RefCell`
//! to use, but not in a way that significantly limits the types of objects one ought to use within it, like a `Cell`.
//! It has one rule: you may only access a reference to the inner item once at a time. This applies to both shared references,
//! and mutable references.
//! 
//! `ScopedCell` works by allowing you access to an exclusive mutable reference to the contained value via [`ScopedCell::borrow`],
//! which you can then only modify or copy from within a closure. You are only allowed to access this value once at a time, 
//! otherwise you will recieve an error for violating `ScopedCell`'s simultaneous access rule.
//! You interact with a `ScopedCell` using similar usage semantics as you would a `Cell`; invoke `borrow` to fetch and observe
//! the inner value like you would with [`std::cell::Cell::get`], and invoke `borrow` as well to update the inner value, like
//! with [`std::cell::Cell::set`].
//! 
//! The limitation of only having access to these borrows from within closure boundaries allows one to much more easily visually
//! verify that multiple simultaneous accesses of the inner data are not being attempted at once; simply keep the bodies of these
//! borrow closures small, and don't invoke any other `ScopedCell` methods within the closure.
//! 
//! `ScopedCell` provides panicking and erroring versions of all its methods, for cases where you for some reason are not sure
//! whether or not you're properly upholding the access rule.
//! 
//! ```
//! use scopedcell::ScopedCell;
//! 
//! let mut scoped_cell = ScopedCell::new(5);
//! scoped_cell.borrow(|inner| *inner *= 10);
//! let result = scoped_cell.borrow(|inner| *inner);
//! assert_eq!(result, 50);
//! ```

use std::{cell::Cell, fmt::Debug};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum ScopedCellError {
    #[error("Attempted to open a ScopedCell while its value is occupied")]
    DataOccupied,
}

pub struct ScopedCell<T> {
    value: Cell<Option<T>>,
}

impl<T> ScopedCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Cell::new(Some(value)),
        }
    }

    pub fn get_and_set<F, U>(&self, f: F) -> U
    where
        F: FnOnce(T) -> (T, U),
    {
        if let Some(value) = self.value.replace(None) {
            let (value, result) = f(value);
            self.value.set(Some(value));
            result
        } else {
            panic!("Attempted to open a ScopedCell while its value is occupied");
        }
    }

    pub fn try_get_and_set<F, U>(&self, f: F) -> Result<U, ScopedCellError>
    where
        F: FnOnce(T) -> (T, U),
    {
        if let Some(value) = self.value.replace(None) {
            let (value, result) = f(value);
            self.value.set(Some(value));
            Ok(result)
        } else {
            Err(ScopedCellError::DataOccupied)
        }
    }

    pub fn morph<F>(&self, f: F)
    where
        F: FnOnce(T) -> T,
    {
        self.get_and_set(|value| (f(value), ()));
    }

    pub fn try_morph<F>(&self, f: F) -> Result<(), ScopedCellError>
    where
        F: FnOnce(T) -> T,
    {
        self.try_get_and_set(|value| (f(value), ()))
    }

    pub fn borrow<U, F>(&self, f: F) -> U
    where
        F: FnOnce(&mut T) -> U,
    {
        self.get_and_set(|mut value| {
            let result = f(&mut value);
            (value, result)
        })
    }

    pub fn try_borrow<U, F>(&self, f: F) -> Result<U, ScopedCellError>
    where
        F: FnOnce(&mut T) -> U,
    {
        self.try_get_and_set(|mut value| {
            let result = f(&mut value);
            (value, result)
        })
    }

    pub fn set(&self, value: T) {
        self.get_and_set(|_| (value, ()));
    }

    pub fn try_set(&self, value: T) -> Result<(), ScopedCellError> {
        self.try_get_and_set(|_| (value, ()))
    }

    pub fn into_inner(self) -> T {
        self.value.into_inner().expect(
            "Attempted to extract the inner value of a ScopedCell while its value is occupied",
        )
    }

    pub fn replace(&self, value: T) -> T {
        self.get_and_set(|inner| (inner, value))
    }

    pub fn try_replace(&self, value: T) -> Result<T, ScopedCellError> {
        self.try_get_and_set(|inner| (inner, value))
    }

    pub fn replace_with<F>(&self, f: F) -> T
    where
        F: FnOnce(&mut T) -> T,
    {
        self.replace(self.get_and_set(|mut inner| (f(&mut inner), inner)))
    }

    pub fn try_replace_with<F>(&self, f: F) -> Result<T, ScopedCellError>
    where
        F: FnOnce(&mut T) -> T,
    {
        let old_value = self.try_get_and_set(|mut inner| (f(&mut inner), inner))?;
        self.try_replace(old_value)
    }

    pub fn take(&self) -> T
    where
        T: Default,
    {
        self.get_and_set(|inner| (inner, T::default()))
    }

    pub fn try_take(&self) -> Result<T, ScopedCellError>
    where
        T: Default,
    {
        self.try_get_and_set(|inner| (inner, T::default()))
    }

    pub fn swap(&self, other: &Self) {
        self.get_and_set(|value| {
            let other_value = other.get_and_set(|other_value| (other_value, value));
            (other_value, ())
        });
    }

    pub fn try_swap(&self, other: &Self) -> Result<(), ScopedCellError> {
        let Some(self_value) = self.value.replace(None) else {
            return Err(ScopedCellError::DataOccupied);
        };
        let Some(other_value) = other.value.replace(None) else {
            self.value.set(Some(self_value));
            return Err(ScopedCellError::DataOccupied);
        };
        other.value.set(Some(self_value));
        self.value.set(Some(other_value));
        Ok(())
    }

    pub fn try_clone(&self) -> Result<Self, ScopedCellError>
    where
        T: Clone,
    {
        self.try_get_and_set(|inner| (inner.clone(), Self::new(inner)))
    }
}

impl<T> std::fmt::Debug for ScopedCell<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_value = self.value.replace(None);
        f.debug_struct("ScopedCell")
            .field("value", &inner_value)
            .finish()?;
        self.value.set(inner_value);
        Ok(())
    }
}

impl<T> Clone for ScopedCell<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        self.get_and_set(|inner| (inner.clone(), Self::new(inner)))
    }
}

impl<T> Default for ScopedCell<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for ScopedCell<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> PartialEq for ScopedCell<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.borrow(|self_inner| other.borrow(|other_inner| self_inner == other_inner))
    }
}

impl<T> Eq for ScopedCell<T> where T: Eq {}

impl<T> PartialOrd for ScopedCell<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.borrow(|self_inner| other.borrow(|other_inner| self_inner.partial_cmp(&other_inner)))
    }
}

impl<T> Ord for ScopedCell<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.borrow(|self_inner| other.borrow(|other_inner| self_inner.cmp(&other_inner)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int() {
        let cell = ScopedCell::new(1);
        cell.borrow(|x| *x += 1);
        assert_eq!(cell.borrow(|x| *x), 2);
    }

    #[test]
    fn test_vec() {
        let cell = ScopedCell::new(vec![1, 2, 3]);
        cell.borrow(|x| {
            x.push(4);
        });
        assert_eq!(cell.borrow(|x| x.len()), 4);
    }

    #[test]
    fn test_struct() {
        #[derive(Clone, Debug, PartialEq)]
        struct Test {
            a: i32,
            b: String,
        }

        let cell = ScopedCell::new(Test {
            a: 1,
            b: "test".to_string(),
        });
        cell.borrow(|x| {
            x.a += 1;
            x.b = "updated".to_string();
        });
        assert_eq!(
            cell.borrow(|x| x.clone()),
            Test {
                a: 2,
                b: "updated".to_string()
            }
        );
    }

    #[test]
    fn test_into_inner() {
        let cell = ScopedCell::new(1);
        assert_eq!(cell.into_inner(), 1);
    }
}

/// ```should_panic
/// use mapcell::MoveCell;
///
/// let cell = MoveCell::new(1);
/// cell.update(|x| cell.update(|x| *x += 1));
/// ```
#[allow(dead_code)]
struct ReentrantTest;
