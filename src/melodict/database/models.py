"""SQLAlchemy 2.0 models for the Melodict Corpus."""

import os

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://melodict:melodict_pass@127.0.0.1:5433/melodict_db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Artist(Base):
    __tablename__ = "artists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    genre: Mapped[str | None] = mapped_column(String(50))
    default_instrument: Mapped[str | None] = mapped_column(String(50))

    sessions: Mapped[list["Session"]] = relationship(back_populates="artist")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    artist_id: Mapped[int] = mapped_column(ForeignKey("artists.id"))
    instrument: Mapped[str | None] = mapped_column(String(50))
    dataset_source: Mapped[str] = mapped_column(String(100))
    bpm: Mapped[float | None] = mapped_column(Float)
    key_signature: Mapped[str | None] = mapped_column(String(20))

    artist: Mapped["Artist"] = relationship(back_populates="sessions")
    phrases: Mapped[list["Phrase"]] = relationship(back_populates="session")


class Phrase(Base):
    __tablename__ = "phrases"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    note_count: Mapped[int] = mapped_column(Integer, index=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    # Stores the raw list of NoteTuples: [[pitch, duration, velocity], ...]
    sequence: Mapped[list] = mapped_column(JSON)

    session: Mapped["Session"] = relationship(back_populates="phrases")


def init_db() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)