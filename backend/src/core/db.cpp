#include "db.hpp"

#include <sqlite3.h>

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace fsd {

namespace {

void checkSqlite(int rc, sqlite3* db) {
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
        const char* msg = db != nullptr ? sqlite3_errmsg(db) : "sqlite error";
        throw std::runtime_error(msg);
    }
}

class Statement {
public:
    Statement(sqlite3* db, const char* sql) : db_(db), stmt_(nullptr) {
        const int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr);
        checkSqlite(rc, db_);
    }

    ~Statement() {
        if (stmt_ != nullptr) {
            sqlite3_finalize(stmt_);
        }
    }

    sqlite3_stmt* get() const { return stmt_; }

private:
    sqlite3* db_;
    sqlite3_stmt* stmt_;
};

} // namespace

Db::Db(std::filesystem::path dbPath) : dbPath_(std::move(dbPath)) {}

void Db::ensureSchema() const {
    sqlite3* db = nullptr;
    const int openRc = sqlite3_open(dbPath_.string().c_str(), &db);
    checkSqlite(openRc, db);

    const char* sql =
        "CREATE TABLE IF NOT EXISTS special_points ("
        "id TEXT PRIMARY KEY,"
        "family TEXT NOT NULL,"
        "point_type TEXT NOT NULL,"
        "k INTEGER NOT NULL,"
        "p INTEGER NOT NULL,"
        "re REAL NOT NULL,"
        "im REAL NOT NULL,"
        "source_mode TEXT NOT NULL,"
        "created_at TEXT NOT NULL"
        ");";

    char* err = nullptr;
    const int execRc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
    if (execRc != SQLITE_OK) {
        std::string msg = err != nullptr ? err : "failed to create schema";
        if (err != nullptr) {
            sqlite3_free(err);
        }
        sqlite3_close(db);
        throw std::runtime_error(msg);
    }

    sqlite3_close(db);
}

void Db::insertSpecialPoint(const SpecialPointRecord& record) const {
    sqlite3* db = nullptr;
    const int openRc = sqlite3_open(dbPath_.string().c_str(), &db);
    checkSqlite(openRc, db);

    const char* sql =
        "INSERT INTO special_points "
        "(id, family, point_type, k, p, re, im, source_mode, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";

    Statement stmt(db, sql);

    checkSqlite(sqlite3_bind_text(stmt.get(), 1, record.id.c_str(), -1, SQLITE_TRANSIENT), db);
    checkSqlite(sqlite3_bind_text(stmt.get(), 2, record.family.c_str(), -1, SQLITE_TRANSIENT), db);
    checkSqlite(sqlite3_bind_text(stmt.get(), 3, record.pointType.c_str(), -1, SQLITE_TRANSIENT), db);
    checkSqlite(sqlite3_bind_int(stmt.get(), 4, record.k), db);
    checkSqlite(sqlite3_bind_int(stmt.get(), 5, record.p), db);
    checkSqlite(sqlite3_bind_double(stmt.get(), 6, record.re), db);
    checkSqlite(sqlite3_bind_double(stmt.get(), 7, record.im), db);
    checkSqlite(sqlite3_bind_text(stmt.get(), 8, record.sourceMode.c_str(), -1, SQLITE_TRANSIENT), db);
    checkSqlite(sqlite3_bind_text(stmt.get(), 9, record.createdAt.c_str(), -1, SQLITE_TRANSIENT), db);

    checkSqlite(sqlite3_step(stmt.get()), db);
    sqlite3_close(db);
}

std::vector<SpecialPointRecord> Db::listSpecialPoints(const std::string& familyFilter, int kFilter, int pFilter) const {
    sqlite3* db = nullptr;
    const int openRc = sqlite3_open(dbPath_.string().c_str(), &db);
    checkSqlite(openRc, db);

    std::string sql =
        "SELECT id, family, point_type, k, p, re, im, source_mode, created_at "
        "FROM special_points ";

    bool hasWhere = false;
    if (!familyFilter.empty()) {
        sql += hasWhere ? "AND " : "WHERE ";
        sql += "family = ? ";
        hasWhere = true;
    }
    if (kFilter >= 0) {
        sql += hasWhere ? "AND " : "WHERE ";
        sql += "k = ? ";
        hasWhere = true;
    }
    if (pFilter >= 0) {
        sql += hasWhere ? "AND " : "WHERE ";
        sql += "p = ? ";
        hasWhere = true;
    }
    sql += "ORDER BY created_at DESC, id DESC;";

    Statement stmt(db, sql.c_str());

    int index = 1;
    if (!familyFilter.empty()) {
        checkSqlite(sqlite3_bind_text(stmt.get(), index++, familyFilter.c_str(), -1, SQLITE_TRANSIENT), db);
    }
    if (kFilter >= 0) {
        checkSqlite(sqlite3_bind_int(stmt.get(), index++, kFilter), db);
    }
    if (pFilter >= 0) {
        checkSqlite(sqlite3_bind_int(stmt.get(), index++, pFilter), db);
    }

    std::vector<SpecialPointRecord> rows;
    while (true) {
        const int rc = sqlite3_step(stmt.get());
        if (rc == SQLITE_DONE) {
            break;
        }
        checkSqlite(rc, db);

        SpecialPointRecord row;
        row.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 0));
        row.family = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 1));
        row.pointType = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 2));
        row.k = sqlite3_column_int(stmt.get(), 3);
        row.p = sqlite3_column_int(stmt.get(), 4);
        row.re = sqlite3_column_double(stmt.get(), 5);
        row.im = sqlite3_column_double(stmt.get(), 6);
        row.sourceMode = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 7));
        row.createdAt = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 8));
        rows.push_back(row);
    }

    sqlite3_close(db);
    return rows;
}

std::string nowIso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowT = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&nowT);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::string makeId() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned long long> dist;
    std::ostringstream ss;
    ss << std::hex << dist(rng) << dist(rng);
    return ss.str();
}

} // namespace fsd
