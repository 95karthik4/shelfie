/** The confirmed library: proof that review decisions outlive the scan. */

import React, { useCallback, useEffect, useState } from 'react';
import { FlatList, RefreshControl, StyleSheet, Text, View } from 'react-native';

import { ApiError, ConfirmedBook, fetchLibrary } from '../api';
import { colors, spacing } from '../theme';
import { Button, Card, ErrorNotice } from '../components/ui';

export function LibraryScreen({ reloadKey }: { reloadKey: number }) {
  const [books, setBooks] = useState<ConfirmedBook[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setBooks(await fetchLibrary());
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load your library.');
    } finally {
      setLoading(false);
    }
  }, []);

  // reloadKey changes whenever a book is confirmed elsewhere in the app, so
  // the library is current when the user switches to it.
  useEffect(() => {
    void load();
  }, [load, reloadKey]);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerText}>
          <Text style={styles.title}>Library</Text>
          <Text style={styles.subtitle}>
            {books.length} {books.length === 1 ? 'book' : 'books'} you confirmed
          </Text>
        </View>
        <Button label="Refresh" variant="secondary" onPress={() => void load()} busy={loading} />
      </View>

      {error ? <ErrorNotice message={error} style={styles.error} /> : null}

      <FlatList
        data={books}
        keyExtractor={(book) => String(book.id)}
        contentContainerStyle={styles.list}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={() => void load()} />}
        ListEmptyComponent={
          loading ? null : (
            <Card>
              <Text style={styles.emptyTitle}>Nothing here yet</Text>
              <Text style={styles.emptyBody}>
                Scan a shelf and confirm the books you want to keep. Confirmed books stay here.
              </Text>
            </Card>
          )
        }
        renderItem={({ item }) => (
          <Card style={styles.row}>
            <Text style={styles.bookTitle}>{item.title}</Text>
            <Text style={styles.bookAuthor}>{item.author ?? 'Unknown author'}</Text>
            <Text style={styles.bookMeta}>
              {item.catalog_id ? `Catalog #${item.catalog_id}` : 'Added manually'} ·{' '}
              {formatDate(item.confirmed_at)}
            </Text>
          </Card>
        )}
      />
    </View>
  );
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleString();
}

const styles = StyleSheet.create({
  container: { flex: 1, paddingHorizontal: spacing.lg, paddingTop: spacing.xxl },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  headerText: { flex: 1 },
  title: { fontSize: 26, fontWeight: '800', color: colors.text },
  subtitle: { fontSize: 14, color: colors.textMuted, marginTop: 2 },
  error: { marginTop: spacing.lg },
  list: { paddingVertical: spacing.lg },
  row: { marginBottom: spacing.md },
  bookTitle: { fontSize: 16, fontWeight: '700', color: colors.text },
  bookAuthor: { fontSize: 14, color: colors.textMuted, marginTop: 2 },
  bookMeta: { fontSize: 12, color: colors.textMuted, marginTop: spacing.sm },
  emptyTitle: { fontSize: 17, fontWeight: '700', color: colors.text },
  emptyBody: { fontSize: 14, color: colors.textMuted, lineHeight: 20, marginTop: spacing.xs },
});
