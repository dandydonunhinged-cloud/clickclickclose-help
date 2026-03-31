"""
CCC Underwriting Database
=========================
SQLite database for lenders, products, and underwriting guidelines.
Powers the deal matching engine and dynamic site stats.

Schema:
  lenders     — wholesale lender profiles
  products    — individual loan products per lender
  guidelines  — underwriting parameters per product (FICO, LTV, DSCR, etc.)
  states      — state availability per lender
  submissions — deal submissions from the website

Usage:
  from underwriting_db import UnderwritingDB
  db = UnderwritingDB()
  db.init()
  db.add_lender(...)
  matches = db.match_deal(...)
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = os.environ.get("CCC_DB_PATH", str(Path(__file__).parent / "ccc_underwriting.db"))


class UnderwritingDB:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self):
        """Create all tables if they don't exist."""
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS lenders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                website TEXT,
                channel TEXT,
                states TEXT,
                broker_comp_type TEXT,
                min_broker_net_worth TEXT,
                notes TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lender_id INTEGER NOT NULL REFERENCES lenders(id),
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                term TEXT,
                min_fico INTEGER,
                max_ltv REAL,
                max_cltv REAL,
                min_dscr REAL,
                max_dti REAL,
                min_loan REAL,
                max_loan REAL,
                property_types TEXT,
                occupancy TEXT,
                interest_only INTEGER DEFAULT 0,
                prepay_penalty TEXT,
                docs_required TEXT,
                notes TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(lender_id, name)
            );

            CREATE TABLE IF NOT EXISTS guidelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL REFERENCES products(id),
                field TEXT NOT NULL,
                value TEXT NOT NULL,
                notes TEXT,
                UNIQUE(product_id, field)
            );

            CREATE TABLE IF NOT EXISTS lender_states (
                lender_id INTEGER NOT NULL REFERENCES lenders(id),
                state TEXT NOT NULL,
                PRIMARY KEY (lender_id, state)
            );

            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                loan_type TEXT,
                txn_type TEXT,
                property_type TEXT,
                state TEXT,
                city TEXT,
                property_value REAL,
                loan_amount REAL,
                down_pct REAL,
                credit_score INTEGER,
                monthly_rent REAL,
                dscr_ratio REAL,
                arv REAL,
                rehab_budget REAL,
                build_cost REAL,
                noi REAL,
                experience TEXT,
                entity_type TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                denied_elsewhere INTEGER DEFAULT 0,
                notes TEXT,
                source TEXT,
                matched_products TEXT,
                status TEXT DEFAULT 'new',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS borrowers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                phone TEXT,
                credit_score INTEGER,
                experience TEXT,
                entity_type TEXT,
                state TEXT,
                is_foreign_national INTEGER DEFAULT 0,
                is_itin INTEGER DEFAULT 0,
                llc_name TEXT,
                notes TEXT,
                deal_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                borrower_id INTEGER NOT NULL REFERENCES borrowers(id),
                loan_type TEXT,
                txn_type TEXT,
                property_type TEXT,
                property_address TEXT,
                state TEXT,
                city TEXT,
                zip TEXT,
                property_value REAL,
                loan_amount REAL,
                down_pct REAL,
                ltv REAL,
                credit_score INTEGER,
                monthly_rent REAL,
                dscr_ratio REAL,
                arv REAL,
                rehab_budget REAL,
                build_cost REAL,
                noi REAL,
                experience TEXT,
                entity_type TEXT,
                entity_name TEXT,
                denied_elsewhere INTEGER DEFAULT 0,
                notes TEXT,
                source TEXT,
                status TEXT DEFAULT 'new',
                matched_lender TEXT,
                matched_product TEXT,
                submitted_to_lender_at TEXT,
                lender_response TEXT,
                approved_at TEXT,
                closing_date TEXT,
                funded_at TEXT,
                funded_amount REAL,
                broker_comp REAL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS deal_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id INTEGER NOT NULL REFERENCES deals(id),
                doc_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                uploaded_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS deal_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id INTEGER NOT NULL REFERENCES deals(id),
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                read INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS deal_status_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id INTEGER NOT NULL REFERENCES deals(id),
                old_status TEXT,
                new_status TEXT NOT NULL,
                changed_by TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS referral_partners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                company TEXT,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                license_type TEXT,
                license_number TEXT,
                license_state TEXT,
                partner_type TEXT,
                comp_structure TEXT,
                comp_pct REAL,
                comp_flat REAL,
                referral_code TEXT UNIQUE,
                portal_password_hash TEXT,
                active INTEGER DEFAULT 1,
                total_referrals INTEGER DEFAULT 0,
                total_funded INTEGER DEFAULT 0,
                total_comp_earned REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS referrals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                partner_id INTEGER NOT NULL REFERENCES referral_partners(id),
                deal_id INTEGER NOT NULL REFERENCES deals(id),
                borrower_name TEXT,
                status TEXT DEFAULT 'referred',
                comp_amount REAL,
                comp_paid INTEGER DEFAULT 0,
                comp_paid_at TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_referrals_partner ON referrals(partner_id);
            CREATE INDEX IF NOT EXISTS idx_referrals_deal ON referrals(deal_id);
            CREATE INDEX IF NOT EXISTS idx_referral_partners_code ON referral_partners(referral_code);
            CREATE INDEX IF NOT EXISTS idx_referral_partners_email ON referral_partners(email);
            CREATE INDEX IF NOT EXISTS idx_deals_borrower ON deals(borrower_id);
            CREATE INDEX IF NOT EXISTS idx_deals_status ON deals(status);
            CREATE INDEX IF NOT EXISTS idx_deal_docs ON deal_documents(deal_id);
            CREATE INDEX IF NOT EXISTS idx_deal_messages ON deal_messages(deal_id);
            CREATE INDEX IF NOT EXISTS idx_borrowers_email ON borrowers(email);
            CREATE INDEX IF NOT EXISTS idx_products_type ON products(type);
            CREATE INDEX IF NOT EXISTS idx_products_lender ON products(lender_id);
            CREATE INDEX IF NOT EXISTS idx_products_active ON products(active);
            CREATE INDEX IF NOT EXISTS idx_lenders_active ON lenders(active);
            CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
        """)
        conn.commit()
        conn.close()

    # ---- Lender CRUD ----

    def add_lender(self, slug, name, website=None, channel=None, states=None,
                   broker_comp_type=None, notes=None):
        conn = self._conn()
        conn.execute("""
            INSERT INTO lenders (slug, name, website, channel, states, broker_comp_type, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                name=excluded.name, website=excluded.website, channel=excluded.channel,
                states=excluded.states, broker_comp_type=excluded.broker_comp_type,
                notes=excluded.notes, updated_at=datetime('now')
        """, (slug, name, website, channel, states, broker_comp_type, notes))
        conn.commit()
        lender_id = conn.execute("SELECT id FROM lenders WHERE slug=?", (slug,)).fetchone()["id"]
        conn.close()
        return lender_id

    def add_product(self, lender_id, name, product_type, term=None, min_fico=None,
                    max_ltv=None, min_dscr=None, max_dti=None, min_loan=None, max_loan=None,
                    property_types=None, occupancy=None, interest_only=False, notes=None):
        conn = self._conn()
        prop_str = json.dumps(property_types) if isinstance(property_types, list) else property_types
        conn.execute("""
            INSERT INTO products (lender_id, name, type, term, min_fico, max_ltv, min_dscr,
                max_dti, min_loan, max_loan, property_types, occupancy, interest_only, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lender_id, name) DO UPDATE SET
                type=excluded.type, term=excluded.term, min_fico=excluded.min_fico,
                max_ltv=excluded.max_ltv, min_dscr=excluded.min_dscr, max_dti=excluded.max_dti,
                min_loan=excluded.min_loan, max_loan=excluded.max_loan,
                property_types=excluded.property_types, occupancy=excluded.occupancy,
                interest_only=excluded.interest_only, notes=excluded.notes,
                updated_at=datetime('now')
        """, (lender_id, name, product_type, term, min_fico, max_ltv, min_dscr,
              max_dti, min_loan, max_loan, prop_str, occupancy, int(interest_only), notes))
        conn.commit()
        pid = conn.execute("SELECT id FROM products WHERE lender_id=? AND name=?",
                           (lender_id, name)).fetchone()["id"]
        conn.close()
        return pid

    # ---- Stats ----

    def get_stats(self):
        conn = self._conn()
        lender_count = conn.execute("SELECT COUNT(*) as c FROM lenders WHERE active=1").fetchone()["c"]
        product_count = conn.execute("SELECT COUNT(*) as c FROM products WHERE active=1").fetchone()["c"]
        types = conn.execute("SELECT DISTINCT type FROM products WHERE active=1").fetchall()
        conn.close()
        return {
            "lenders": lender_count,
            "products": product_count,
            "types": len(types),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    # ---- Deal Matching ----

    def match_deal(self, loan_type, property_type=None, txn_type=None,
                   credit_score=None, ltv=None, dscr_ratio=None, state=None):
        """Match a deal against all active products. Returns ranked list."""
        conn = self._conn()

        type_map = {
            'dscr': ('dscr', 'portfolio', 'long-term rental'),
            'str': ('dscr',),
            'flip': ('bridge', 'short-term bridge'),
            'bridge': ('bridge', 'short-term bridge', 'bridge-to-perm'),
            'construction': ('construction', 'construction-to-perm'),
            'portfolio': ('portfolio', 'blanket', 'dscr'),
            'multifamily': ('multifamily', 'commercial', 'investor'),
            'commercial': ('commercial', 'multifamily'),
            'sba': ('sba', 'commercial'),
        }
        target_types = type_map.get(loan_type, (loan_type,))

        # Get all active products with lender info
        rows = conn.execute("""
            SELECT p.*, l.name as lender_name, l.website as lender_website,
                   l.channel as lender_channel, l.states as lender_states
            FROM products p
            JOIN lenders l ON p.lender_id = l.id
            WHERE p.active = 1 AND l.active = 1
        """).fetchall()
        conn.close()

        matches = []
        for row in rows:
            score = 0
            qualifies = False

            # Type match
            for t in target_types:
                if t.lower() in row["type"].lower():
                    qualifies = True
                    score += 50
                    break

            if not qualifies:
                continue

            # Credit score check
            if credit_score and row["min_fico"]:
                if credit_score >= row["min_fico"]:
                    score += 20
                else:
                    continue  # Hard disqualifier

            # LTV check
            if ltv and row["max_ltv"]:
                if ltv <= row["max_ltv"]:
                    score += 15
                else:
                    continue  # Hard disqualifier

            # DSCR check
            if dscr_ratio and row["min_dscr"]:
                if dscr_ratio >= row["min_dscr"]:
                    score += 15
                else:
                    score -= 20  # Soft penalty, some lenders have no-ratio

            # Transaction type bonus
            if txn_type:
                name_l = row["name"].lower()
                if txn_type == "cashout" and ("cash-out" in name_l or "cash out" in name_l):
                    score += 15
                elif txn_type == "refinance" and ("refi" in name_l or "refinance" in name_l):
                    score += 15
                elif txn_type == "purchase" and "purchase" in name_l:
                    score += 10

            # STR bonus
            if loan_type == "str":
                name_l = row["name"].lower()
                if "str" in name_l or "short-term" in name_l or "airbnb" in name_l:
                    score += 25

            matches.append({
                "lender": row["lender_name"],
                "lender_website": row["lender_website"],
                "lender_channel": row["lender_channel"],
                "product": row["name"],
                "type": row["type"],
                "term": row["term"],
                "min_fico": row["min_fico"],
                "max_ltv": row["max_ltv"],
                "min_dscr": row["min_dscr"],
                "score": score
            })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    # ---- Submissions ----

    def save_submission(self, data):
        conn = self._conn()
        conn.execute("""
            INSERT INTO submissions (loan_type, txn_type, property_type, state, city,
                property_value, loan_amount, down_pct, credit_score, monthly_rent,
                dscr_ratio, arv, rehab_budget, build_cost, noi, experience, entity_type,
                name, email, phone, denied_elsewhere, notes, source, matched_products)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("loanType"), data.get("txn"), data.get("propType"),
            data.get("state"), data.get("city"), data.get("value"),
            data.get("loan_amount"), data.get("down"), data.get("credit"),
            data.get("rent"), data.get("dscr"), data.get("arv"),
            data.get("rehab"), data.get("buildCost"), data.get("noi"),
            data.get("experience"), data.get("entity"), data.get("name"),
            data.get("email"), data.get("phone"),
            1 if data.get("denied") == "yes" else 0,
            data.get("notes"), data.get("source"),
            json.dumps(data.get("matched_products", []))
        ))
        conn.commit()
        sub_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return sub_id


# Singleton
_db = None
def get_db():
    global _db
    if _db is None:
        _db = UnderwritingDB()
        _db.init()
    return _db
