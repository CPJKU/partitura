#!/usr/bin/env python

import numpy as np
import socket
import struct
import json
import logging
import traceback

# use an 8 byte unsigned integer for the length of a message
N_BYTES_MSG_LEN = 8
MSG_LEN_TYPE = '!Q'


def read_varlen_msg(connection):
    logger = logging.getLogger(__name__)

    # read how long the incoming message will be
    try:
        raw_len = r''
        while len(raw_len) < N_BYTES_MSG_LEN:
            new = connection.recv(N_BYTES_MSG_LEN - len(raw_len))
            if len(new) == 0:
                break
            raw_len += new
    except socket.error:
        logger.info('connection error while reading message length')
        return None

    if len(raw_len) < N_BYTES_MSG_LEN:
        logger.info('connection closed before message length could be read')
        return None

    try:
        msg_len = struct.unpack(MSG_LEN_TYPE, raw_len)[0]
        logger.debug('announced message length: {}'.format(msg_len))
    except:
        logger.warning('could not interpret message length')
        return None

    # read the incoming message
    try:
        msg = r''
        while len(msg) < msg_len:
            new = connection.recv(msg_len - len(msg))
            if len(new) == 0:
                break
            msg += new
    except socket.error:
        logger.info('connection error while reading message')
        return None

    if len(msg) < msg_len:
        logger.info(
            'connection closed before message could be read {}/{} bytes'.format(len(msg), msg_len))
        return None

    logger.debug('received message')
    return msg


def write_varlen_msg(connection, msg):
    logger = logging.getLogger(__name__)

    # TODO: this could fail for *very* large objects
    try:
        raw_len = struct.pack(MSG_LEN_TYPE, len(msg))
    except TypeError:
        logger.warning('cannot send data of type {}'.format(type(msg)))
        return False

    try:
        connection.sendall(raw_len)
        connection.sendall(msg)
    except Exception:
        logger.warning('lost connection while sending data')
        raise

    return True


def read_json(connection):
    logger = logging.getLogger(__name__)
    # interpret the message as json
    try:
        return json.loads(read_varlen_msg(connection))
    except:
        logger.warning('could not interpret message')
        return None


def write_json(connection, content):
    logger = logging.getLogger(__name__)
    try:
        msg = json.dumps(content)
    except:
        logger.warning('failed to create a json representation of the data')
        return False

    return write_varlen_msg(msg)


def run_server(server_address, func, only_allow_from=None):
    logger = logging.getLogger(__name__)
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to the port
    # server_address = ('localhost', port)

    logger.info('starting up on {} port {}'.format(*server_address))

    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    try:
        while True:
            # Wait for a connection
            logger.info('waiting for a connection')

            connection, client_address = sock.accept()

            logger.info('connection from {}'.format(client_address))

            if only_allow_from and (client_address[0] not in only_allow_from):
                logger.info('rejecting unauthorized connection from {}'.format(
                    client_address[0]))
                connection.close()
                continue
            else:
                logger.info('client is authorized: {}'.format(
                    client_address[0]))

            try:
                while True:
                    logger.info('waiting for next function call from client')

                    data = read_varlen_msg(connection)

                    if data is None:
                        logger.info('closing connection')
                        break

                    try:
                        result = func(data)
                    except Exception:
                        logger.warning(
                            'failed to evaluate function on data from client')
                        traceback.print_exc()
                        break
                        # raise

                    success = write_varlen_msg(connection, result)
                    if not success:
                        logger.warning('failed send function result to client')
                        break
                    else:
                        logger.info('function applied successfully')

            finally:
                # Clean up the connection
                connection.close()
                logger.info('connection closed')

    except KeyboardInterrupt:
        logger.info('received KeyboardInterrupt, closing socket')
        sock.close()


def main():

    # this is the function that the server will apply to any received data, to
    # return its result to the client:

    def mysum(x):
        """
        Sum all elements in a (nested) list of numbers x
        """
        return np.sum(x)

    run_server(mysum)


if __name__ == '__main__':
    main()
