import { DataSourceOptions } from "typeorm";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TypeORMVectorStore } from "langchain/vectorstores/typeorm";

// First, follow set-up instructions at
// https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/typeorm

export const run = async () => {
  const args = {
    postgresConnectionOptions: {
      type: "postgres",
      host: "localhost",
      port: 5432,
      username: "myuser",
      password: "!ChangeMe!",
      database: "api",
    } as DataSourceOptions,
  };

  const typeormVectorStore = await TypeORMVectorStore.fromDataSource(
    new OpenAIEmbeddings(),
    args
  );

  await typeormVectorStore.initTable();

  const docHello = {
    pageContent: "hello",
    metadata: { a: 1 },
    sourceName: "hello.txt",
  };
  const docHi = { pageContent: "hi", metadata: { a: 1 }, sourceName: "hi.txt" };
  const docCat = {
    pageContent: "Cat drinks milk",
    metadata: { a: 2, sourceName: "cat.txt" },
  };

  await typeormVectorStore.addDocuments([
    docHello,
    docHi,
    docCat,
    { pageContent: "what's this", metadata: { a: 2, sourceName: "what.txt" } },
  ]);

  const results = await typeormVectorStore.similaritySearch("hello", 2);

  console.log(results);
};
