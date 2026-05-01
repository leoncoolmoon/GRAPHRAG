import unittest
from unittest.mock import MagicMock, patch
from main import process_text_file, chunk_process, Entity, Relationship

class TestGraphRAGFixes(unittest.TestCase):
    def test_process_text_file(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        result = process_text_file(text, page=1)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], "Paragraph 1.")
        self.assertEqual(result[0]['metadata']['page'], 1)
        self.assertEqual(result[0]['metadata']['paragraph_index'], 0)

    @patch('main.graph_rag')
    def test_chunk_process_logic(self, mock_graph_rag):
        # Mock dependencies
        mock_graph_rag.neo4j_manager = MagicMock()
        mock_graph_rag.ollama_client = MagicMock()
        mock_graph_rag.text_processor = MagicMock()

        mock_graph_rag.ollama_client.embed.return_value = [0.1, 0.2]
        # Return a fresh Entity each time to avoid sharing state if necessary
        def mock_extract(text):
            return ([Entity(name="Entity1", type="Person", properties={})],
                    [Relationship(source="Entity1", target="Entity2", type="WorksAt", properties={})])

        mock_graph_rag.text_processor.extract_entities_and_relations.side_effect = mock_extract

        paragraphs = [
            {"text": "P1", "metadata": {"page": 0, "paragraph_index": 0}},
            {"text": "P2", "metadata": {"page": 0, "paragraph_index": 1}},
            {"text": "P3", "metadata": {"page": 0, "paragraph_index": 2}}
        ]

        from main import chunk_process
        success = chunk_process(paragraphs, "doc123", rebuilder=True)

        self.assertTrue(success)
        self.assertTrue(mock_graph_rag.neo4j_manager.store_chunk.called)
        self.assertTrue(mock_graph_rag.neo4j_manager.store_entities.called)

        # Verify the Entity properties update in chunk_process
        # store_entities was called for i=0 and i=2
        # For i=2, window[1] is paragraphs[2], so paragraph_index=2
        last_call_args = mock_graph_rag.neo4j_manager.store_entities.call_args_list[-1]
        entities_arg = last_call_args[0][0]
        self.assertTrue(hasattr(entities_arg[0], 'properties'))
        self.assertEqual(entities_arg[0].properties['paragraph_index'], 2)

if __name__ == '__main__':
    unittest.main()
