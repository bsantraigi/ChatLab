# Setup the sessions.db file for the sqlite3 database

# Create the database
sqlite3 sessions.db << EOF
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    session_data TEXT,
    created INTEGER
);
EOF

# Create the index

# sqlite3 sessions.db << EOF
# CREATE INDEX last_access_idx ON sessions(last_access);
# EOF

