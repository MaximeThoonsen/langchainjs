import { Metadata } from "@opensearch-project/opensearch/api/types.js";
import {
  DataSource as DataSourceT,
  DataSourceOptions,
  EntitySchema,
} from "typeorm";
import { VectorStore } from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { CharacterTextSplitter } from "../text_splitter.js";
import {
  getSourceNameFromDocument,
  getSourceTypeFromDocument,
  getUniqueIDFromDocument,
} from "../util/document_utils.js";

export interface TypeORMVectorStoreArgs {
  postgresConnectionOptions: DataSourceOptions;
  tableName?: string;
  filter?: Metadata;
}

export class TypeORMVectorStoreDocument extends Document {
  embedding: string;
}

const defaultDocumentTableName = "documents";

export class TypeORMVectorStore extends VectorStore {
  declare FilterType: Metadata;

  tableName: string;

  documentEntity: EntitySchema;

  filter?: Metadata;

  appDataSource: DataSourceT;

  _verbose?: boolean;

  private constructor(embeddings: Embeddings, fields: TypeORMVectorStoreArgs) {
    super(embeddings, fields);
    this.tableName = fields.tableName || defaultDocumentTableName;
    this.filter = fields.filter;
  }

  static async fromDataSource(
    embeddings: Embeddings,
    fields: TypeORMVectorStoreArgs
  ): Promise<TypeORMVectorStore> {
    const TypeORMDocumentEntity = new EntitySchema<TypeORMVectorStoreDocument>({
      name: fields.tableName ?? defaultDocumentTableName,
      columns: {
        id: {
          type: String,
          primary: true,
        },
        pageContent: {
          type: String,
        },
        metadata: {
          type: "jsonb",
        },
        sourceType: {
          type: String,
        },
        sourceName: {
          type: String,
        },
        hash: {
          type: String,
        },
        embedding: {
          type: String,
        },
      },
    });

    const appDataSource = new DataSourceT({
      entities: [TypeORMDocumentEntity],
      ...fields.postgresConnectionOptions,
    });

    const postgresqlVectorStore = new TypeORMVectorStore(embeddings, fields);
    postgresqlVectorStore.appDataSource = appDataSource;
    postgresqlVectorStore.documentEntity = TypeORMDocumentEntity;

    if (!postgresqlVectorStore.appDataSource.isInitialized) {
      await postgresqlVectorStore.appDataSource.initialize();
    }

    postgresqlVectorStore._verbose =
      typeof process !== "undefined"
        ? // eslint-disable-next-line no-process-env
          process.env?.LANGCHAIN_VERBOSE !== undefined
        : false;

    return postgresqlVectorStore;
  }

  async addDocuments(documents: Document[]): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents
    );
  }

  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    const { createHash } = await import("node:crypto");
    const rows = vectors.map((embedding, idx) => {
      const embeddingString = `[${embedding.join(",")}]`;
      const hash = createHash("sha1");
      const documentRow = {
        pageContent: documents[idx].pageContent,
        embedding: embeddingString,
        metadata: documents[idx].metadata,
        id: documents[idx].id ?? getUniqueIDFromDocument(documents[idx]),
        sourceType: getSourceTypeFromDocument(documents[idx]),
        sourceName: getSourceNameFromDocument(documents[idx]),
        hash:
          documents[idx].hash ??
          hash.update(documents[idx].pageContent).digest("hex"),
      } as TypeORMVectorStoreDocument;

      return documentRow;
    });

    const documentRepository = this.appDataSource.getRepository(
      this.documentEntity
    );

    // For a given document, we delete all documents in the database that have not the same id as
    // the ones we are trying to upsert because it means that the split of the file is not the same anymore
    for (const doc of rows) {
      if (!doc.sourceName || !doc.sourceType) {
        continue;
      }
      // Search for existing document with sourceName and type
      const documentsInDatabase = await documentRepository.find({
        where: {
          sourceName: doc.sourceName,
          sourceType: doc.sourceType,
        },
      });

      const documentsThatWillBeUpserted = rows.filter(
        (d) => d.sourceName === doc.sourceName
      );
      const idsDocumentsThatWillBeUpserted = documentsThatWillBeUpserted.map(
        (d) => d.id
      );

      const idsDocumentsThatWeShouldDeleted = documentsInDatabase
        .filter((d) => !idsDocumentsThatWillBeUpserted.includes(d.id))
        .map((d) => d.id);

      this.verboseConsoleLog(
        `Ids has changed for document ${doc.sourceName} and ${doc.sourceType}, action: delete this ids: ${idsDocumentsThatWeShouldDeleted}`
      );

      await documentRepository.delete({
        id: {
          in: idsDocumentsThatWeShouldDeleted,
        },
      });
    }

    const documentsToUpsert = [];
    for (const row of rows) {
      // Search for existing document with same sourceName and sourcetype
      const documentsInDatabase = (await documentRepository.findOne({
        where: {
          id: row.id,
        },
      })) as Document;

      if (documentsInDatabase) {
        if (documentsInDatabase.hash === row.hash) {
          this.verboseConsoleLog(
            `Same hash was found for id ${row.id}, action: ignore`
          );
        } else {
          documentsToUpsert.push(row);
          this.verboseConsoleLog(
            `Different hash was found for id ${row.id}, action: update embedding`
          );
        }
      } else {
        documentsToUpsert.push(row);
        this.verboseConsoleLog(
          `No document was found for id ${row.id}, action: create`
        );
      }
    }

    const chunkSize = 500;
    for (let i = 0; i < documentsToUpsert.length; i += chunkSize) {
      const chunk = documentsToUpsert.slice(i, i + chunkSize);

      try {
        await documentRepository.upsert(chunk, [`id`]);
      } catch (e) {
        console.error(e);
        throw new Error(`Error inserting: ${chunk[0].pageContent}`);
      }
    }
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"]
  ): Promise<[TypeORMVectorStoreDocument, number][]> {
    const embeddingString = `'[${query.join(",")}]'`;
    const _filter = filter ?? "'{}'";
    const queryString = `
      SELECT *, embedding <=> ${embeddingString}::vector as "_distance" 
      FROM ${this.tableName}
      WHERE metadata @> ${_filter}
      ORDER BY "_distance" ASC
      LIMIT ${k};
    `;
    const documents = await this.appDataSource.query(queryString);

    const results = [] as [TypeORMVectorStoreDocument, number][];
    for (const doc of documents) {
      if (doc._distance != null && doc.pageContent != null) {
        const document = new Document(doc) as TypeORMVectorStoreDocument;
        document.id = doc.id;
        results.push([document, doc._distance]);
      }
    }

    return results;
  }

  async initTable(): Promise<void> {
    await this.appDataSource.query("CREATE EXTENSION IF NOT EXISTS vector;");

    await this.appDataSource.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        id text PRIMARY KEY,
        "pageContent" text,
        metadata jsonb,
        "sourceType" text,
        "sourceName" text,
        hash text,
        embedding vector
      );
    `);
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig: TypeORMVectorStoreArgs
  ): Promise<TypeORMVectorStore> {
    const docs = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    const splitter = new CharacterTextSplitter();

    const splittedDocs = await splitter.splitDocuments(docs);

    return TypeORMVectorStore.fromDocuments(splittedDocs, embeddings, dbConfig);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: TypeORMVectorStoreArgs
  ): Promise<TypeORMVectorStore> {
    const instance = await TypeORMVectorStore.fromDataSource(
      embeddings,
      dbConfig
    );
    await instance.addDocuments(docs);

    return instance;
  }

  static async fromExistingIndex(
    embeddings: Embeddings,
    dbConfig: TypeORMVectorStoreArgs
  ): Promise<TypeORMVectorStore> {
    const instance = await TypeORMVectorStore.fromDataSource(
      embeddings,
      dbConfig
    );
    return instance;
  }

  private verboseConsoleLog(message: string): void {
    if (this._verbose) {
      console.log(message);
    }
  }
}
