/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the action functions to either generate new events, determine a protocol
 * or read data from application or write data to application
 *
 * @file smbus_action.c
 *
 */

#include <string.h>
#include <stdio.h>
#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_action.h"
#include "smbus_event.h"
#include "smbus_hardware_access.h"

/******************************************************************************
*
* @brief    Clears all actions that have been raised against the specified instance.
*
*****************************************************************************/
void vSMBusClearAction( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        pxSMBusInstance->ulAction = 0;
    }
}

/******************************************************************************
*
* @brief    Sets a specific actions against the specified instance
*
*****************************************************************************/
void vSMBusAction( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint32_t ulAnyAction )
{
    if( NULL != pxSMBusInstance )
    {
        pxSMBusInstance->ulAction |= ulAnyAction;
    }
}

/******************************************************************************
*
* @brief    Resets all variables used during an SMBus message transaction back
*           to their default values
*           Resets the IP's Descriptor and RX FIFOs
*
*****************************************************************************/
void vSMBusHandleActionResetAllData( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {   
        pxSMBusInstance->ucCommand                      = SMBUS_COMMAND_INVALID;
        pxSMBusInstance->xProtocol                      = SMBUS_PROTOCOL_NONE;
        pxSMBusInstance->usSendDataSize                 = 0;
        pxSMBusInstance->usSendIndex                    = 0;
        pxSMBusInstance->usReceiveIndex                 = 0;
        pxSMBusInstance->usExpectedByteCount            = 0;
        pxSMBusInstance->ucNewDeviceSlaveAddress        = 0;
        pxSMBusInstance->ucNackSent                     = SMBUS_FALSE;
        pxSMBusInstance->usDescriptorsSent              = 0;
        pxSMBusInstance->ucPECSent                      = SMBUS_FALSE;
        pxSMBusInstance->ucFifoEmptyWhileInDoneCount    = 0;
        pxSMBusInstance->ucUDIDMatchedInstance          = SMBUS_INVALID_INSTANCE;

        SMBUS_PROFILE_TYPE* pxSMBusProfile = pxSMBusInstance->pxSMBusProfile;

        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG, 
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                        ( uint32_t )pxSMBusInstance->ucThisInstanceNumber, __LINE__ );

        if( pxSMBusInstance->ucThisInstanceNumber == pxSMBusProfile->ucInstanceInPlay ) /* This is controller */
        {
            pxSMBusProfile->ucInstanceInPlay = SMBUS_INVALID_INSTANCE;

            /* Empty any RX queues */
            while( !ulSMBusHWReadCtrlRxFifoStatusEmpty( pxSMBusProfile ) )
            {
                ulSMBusHWReadCtrlRxFifoPayload( pxSMBusProfile );
            }

            vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );
            vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
        }
        else /* This is a target */
        {
            pxSMBusProfile->ucActiveTargetInstance = SMBUS_INVALID_INSTANCE;
            /* Empty any RX queues */
            while( !ulSMBusHWReadTgtRxFifoStatusEmpty( pxSMBusProfile ) )
            {
                ulSMBusHWReadTgtRxFifoPayload( pxSMBusProfile );
            }

            vSMBusHWWriteTgtRxFifoReset( pxSMBusProfile, 1 );
        }
    }
}

/******************************************************************************
*
* @brief    Generates an E_IS_PEC_REQUIRED event against the specified instance
*
*****************************************************************************/
void vSMBusHandleActionCreateEventIsPECRequired( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusGenerateEvent_E_IS_PEC_REQUIRED( pxSMBusInstance );
    }
}

/******************************************************************************
*
* @brief    Generates an E_SEND_NEXT_BYTE event against the specified instance
*
*****************************************************************************/
void vSMBusHandleActionCreateEventSendNextByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusGenerateEvent_E_SEND_NEXT_BYTE( pxSMBusInstance );
    }
}

/******************************************************************************
*
* @brief    Reads the command byte from the Target RX FIFO, calls the callback function
*           for the specified instance to get the SMBus protocol associated with the
*           command byte and stores that protocol in the instance structure
*
*****************************************************************************/
void vSMBusHandleActionGetProtocol( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    SMBus_Command_Protocol_Type xTempProtocol = SMBUS_PROTOCOL_NONE;

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnGetProtocol ) )
    {
        if( SMBUS_RX_FIFO_IS_EMPTY != ulSMBusHWReadTgtRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
        {
            uint8_t ucCommand = ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
            pxSMBusInstance->ucCommand = ucCommand;
            pxSMBusInstance->pFnGetProtocol( ucCommand, &xTempProtocol );
            pxSMBusInstance->xProtocol = xTempProtocol;
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_INFO, 
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_PROTOCOL,
                            ( uint32_t )ucCommand, ( uint32_t )pxSMBusInstance->xProtocol );
        }
        else
        {
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, 
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
        }
    }
}

/******************************************************************************
*
* @brief    Reads the command byte from the Target RX FIFO for the ARP instance,
*           determines the ARP protocol associated with the command byte,
*           stores that protocol in the instance structure
*
*****************************************************************************/
uint8_t ucSMBusHandleActionGetARPProtocol( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    uint8_t ucReturnCode = SMBUS_ACTION_ARP_PROTOCOL_DETERMINED;
    uint8_t ucARPCommand = 0;

    if( NULL != pxSMBusInstance )
    {
        if( SMBUS_RX_FIFO_IS_EMPTY != ulSMBusHWReadTgtRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
        {
            ucARPCommand = ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
            
            pxSMBusInstance->ucCommand = ucARPCommand;

            switch( ucARPCommand )
            {
            case 0x01:
                pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP;
                break;

            case 0x02:
                pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_RESET_DEVICE;
                break;

            case 0x03:
                pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_GET_UDID;
                break;

            case 0x04:
                pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS;
                break;

            case 0x00:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09:
            case 0x0a:
            case 0x0b:
            case 0x0c:
            case 0x0d:
            case 0x0e:
            case 0x0f:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1a:
            case 0x1b:
            case 0x1c:
            case 0x1d:
            case 0x1e:
            case 0x1f:
                /* Reserved */
                ucReturnCode = SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED;
                break;
            
            default:
                /* Anything else is a directed ARP command */
                if( SMBUS_ARP_UDID_DIRECTED_COMMAND == ( ucARPCommand & SMBUS_ARP_DIRECTED_COMMAND_MASK ) )
                {
                    pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED;
                }
                else
                {
                    pxSMBusInstance->xProtocol = SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED;
                }

                pxSMBusInstance->ucMatchedSMBusAddress = ( ucARPCommand & SMBUS_ARP_DIRECTED_COMMAND_ADDRESS_MASK ) >> 1;
                break;
            }
        }
        else
        {
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, 
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
            ucReturnCode = SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED;
        }       
    }
    else
    {
        ucReturnCode = SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED;
    }
    
    return( ucReturnCode );
}

/******************************************************************************
*
* @brief    Checks if a callback function to read data from the application software
*           for the specified instance is present.
*           If present it is called for the current SMBus command.
*           The callback function must return the data and size of data for the command.
*
*****************************************************************************/
void vSMBusHandleActionGetDataFromApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    uint8_t     ucTempSendData[SMBUS_DATA_SIZE_MAX] = { 0 };
    uint16_t    usTempSendDataSize                  = 0;

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnGetData ) )
    {
        pxSMBusInstance->pFnGetData( pxSMBusInstance->ucCommand, ucTempSendData, &usTempSendDataSize );

        if( SMBUS_DATA_SIZE_MAX >= usTempSendDataSize )
        {
            memcpy( pxSMBusInstance->ucSendData, ucTempSendData, usTempSendDataSize );

            switch( pxSMBusInstance->xProtocol )
            {
            case SMBUS_PROTOCOL_READ_64:
                pxSMBusInstance->usSendDataSize = 8;
                break;
            case SMBUS_PROTOCOL_READ_32:
                pxSMBusInstance->usSendDataSize = 4;
                break;
            case SMBUS_PROTOCOL_READ_WORD:
                pxSMBusInstance->usSendDataSize = 2;
                break;
            case SMBUS_PROTOCOL_READ_BYTE:
                pxSMBusInstance->usSendDataSize = 1;
                break;
            case SMBUS_PROTOCOL_PROCESS_CALL:
                pxSMBusInstance->usSendDataSize = 2;
                break;
            case SMBUS_PROTOCOL_RECEIVE_BYTE:
                pxSMBusInstance->usSendDataSize = 1;
                break;
            default:
                pxSMBusInstance->usSendDataSize = usTempSendDataSize;
                break;
            }
        }
        else
        {
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
        }
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to write data to the application software
*           for the specified instance is present.
*           If present it is called for the current SMBus command.
*           The callback function write the command, data and size of data and transaction ID.
*
*****************************************************************************/
void vSMBusHandleActionWriteDataToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucTransactionID )
{
    SMBUS_PROFILE_TYPE* pxSMBusProfile = NULL;
	uint8_t	ucTempReceivedData[SMBUS_DATA_SIZE_MAX] = { 0 };

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnWriteData ) )
    {
        pxSMBusProfile = pxSMBusInstance->pxSMBusProfile;

        /* Only return the tranaction ID for Controller */
        if( pxSMBusInstance->ucThisInstanceNumber != pxSMBusProfile->ucInstanceInPlay )
        {
            ucTransactionID = 0;
        }

        memcpy( ucTempReceivedData, pxSMBusInstance->ucReceivedData, pxSMBusInstance->usExpectedByteCount );

        pxSMBusInstance->pFnWriteData( pxSMBusInstance->ucCommand, ucTempReceivedData,
                                        pxSMBusInstance->usExpectedByteCount, ucTransactionID );
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to announce the result of the SMBus transaction
*           for the specified instance is present.
*           If present the callback is called.
*           The callback function will include the command, transaction ID and the result
*
*****************************************************************************/
void vSMBusHandleActionAnnounceResultToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, 
                                                        uint8_t ucTransactionID, uint32_t ulStatus )
{
    SMBUS_PROFILE_TYPE* pxSMBusProfile = NULL;

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnAnnounceResult ) )
    {
        pxSMBusProfile = pxSMBusInstance->pxSMBusProfile;

        /* Only return the tranaction ID for Controller */
        if( pxSMBusInstance->ucThisInstanceNumber != pxSMBusProfile->ucInstanceInPlay )
        {
            ucTransactionID = 0;
        }
        pxSMBusInstance->pFnAnnounceResult( pxSMBusInstance->ucCommand, ucTransactionID, ulStatus );
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to announce that an ARP Assign Address
*           for the specified instance is present.
*           If present the callback is called.
*           The callback function will include the newly assigned address
*
*****************************************************************************/
void vSMBusHandleActionNotifyAddressChangeToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, 
                                                                uint8_t ucTransactionID )
{
    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnArpAddressChange ) )
    {
        pxSMBusInstance->pFnArpAddressChange( pxSMBusInstance->ucSMBusAddress );
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to announce an SMBuss Error
*           for the specified instance is present.
*           If present the callback is called.
*           The callback function will include the error type
*
*****************************************************************************/
void vSMBusHandleActionBusError( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucError )
{
    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnBusError ) )
    {
        pxSMBusInstance->pFnBusError( ucError );
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to announce an SMBuss Warning
*           for the specified instance is present.
*           If present the callback is called.
*           The callback function will include the warning type
*
*****************************************************************************/
void vSMBusHandleActionBusWarning( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucWarning )
{
    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnBusWarning ) )
    {
        pxSMBusInstance->pFnBusWarning( ucWarning );
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to read data from the application software
*           for the specified instance is present.
*           If present it is called for the current I2C transaction.
*           The callback function must return the data and size of data for the transaction.
*
*****************************************************************************/
void vSMBusHandleActionGetI2CDataFromApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    uint8_t     ucTempSendData[SMBUS_DATA_SIZE_MAX] = { 0 };
    uint16_t    usTempSendDataSize                  = 0;

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnI2CGetData ) )
    {
        pxSMBusInstance->pFnI2CGetData( ucTempSendData, &usTempSendDataSize );

        if( SMBUS_DATA_SIZE_MAX >= usTempSendDataSize )
        {
            memcpy( pxSMBusInstance->ucSendData, ucTempSendData, usTempSendDataSize );
            pxSMBusInstance->usSendDataSize = usTempSendDataSize;
        }
        else
        {
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
        }
    }
}

/******************************************************************************
*
* @brief    Checks if a callback function to write data to the application software
*           for the specified instance is present.
*           If present it is called for the current I2C transaction.
*           The callback function write the data and size of data and transaction ID.
*
*****************************************************************************/
void vSMBusHandleActionWriteI2CDataToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
	uint8_t	ucTempReceivedData[SMBUS_DATA_SIZE_MAX] = { 0 };

    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnI2CWriteData ) )
    {
        memcpy( ucTempReceivedData, pxSMBusInstance->ucReceivedData, pxSMBusInstance->usReceiveIndex );

        pxSMBusInstance->pFnI2CWriteData( ucTempReceivedData,
                                        pxSMBusInstance->usReceiveIndex );
    }
}


/******************************************************************************
*
* @brief    Checks if a callback function to write data to the application software
*           for the specified instance is present.
*           If present it is called for the current I2C transaction.
*           The callback function write the data and size of data and transaction ID.
*
*****************************************************************************/
void vSMBusHandleActionAnnounceI2CResultToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint32_t ulStatus )
{
    if( ( NULL != pxSMBusInstance ) &&
        ( NULL != pxSMBusInstance->pFnI2CAnnounceResult ) )
    {
        pxSMBusInstance->pFnI2CAnnounceResult( ulStatus );
    }
}
