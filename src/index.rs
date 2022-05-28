use crate::error::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Index {
    pub id: u8,
    pub dim: u64,
    pub plev: u64,
    pub tags: String,
}

impl Index {
    pub fn new(dim: u64) -> Index {
        Index {
            id: rand::random(),
            dim,
            plev: 0,
            tags: "".to_owned(),
        }
    }

    pub fn with_plev(mut self, pl: u64) -> Index {
        self.plev = pl;
        self
    }

    pub fn add_tag(mut self, tag: &str) -> Index {
        if self.tags.is_empty() {
            self.tags = tag.to_owned();
        } else {
            self.tags = format!("{}|{}", self.tags, tag);
        }
        self
    }

    pub fn prime(&mut self) {
        self.plev += 1;
    }

    pub fn prime_incr(&mut self, pl: u64) {
        self.plev += pl;
    }

    pub fn find_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }
}

#[derive(Clone, Debug)]
pub struct IndexVal {
    pub index: Index,
    pub val: u64,
}

impl IndexVal {
    pub fn new(index: &Index, val: u64) -> Result<IndexVal> {
        if val < index.dim {
            Ok(IndexVal {
                index: index.clone(),
                val,
            })
        } else {
            Err(TenRustError::IndexOutOfBounds("IndexVal::new"))
        }
    }
}

#[cfg(test)]
mod index_tests {
    use crate::index::Index;

    #[test]
    fn basic_setup() {
        let i = Index::new(2);
        assert_eq!(i.dim, 2);
    }

    #[test]
    fn setup_with_primes_tags() {
        let mut i = Index::new(2).with_plev(1).add_tag("index");
        assert!(i.find_tag("index"));
        assert!(!i.find_tag("Index"));
        assert_eq!(i.plev, 1);

        i.prime();
        assert_eq!(i.plev, 2);
        i.prime_incr(2);
        assert_eq!(i.plev, 4);
    }
}

#[cfg(test)]
mod indexval_tests {
    use crate::index::Index;
    use crate::index::IndexVal;
    #[test]
    fn setup_indexval() {
        let i = Index::new(2);
        let ival = IndexVal::new(&i, 0);
        assert!(ival.is_ok());
        let ival_unwrapped = ival.unwrap();
        assert_eq!(ival_unwrapped.index.id, i.id);
    }

    #[test]
    fn set_indexval_fail() {
        let i = Index::new(2);
        let wrong_ival = IndexVal::new(&i, 5);
        assert!(wrong_ival.is_err());
    }
}
